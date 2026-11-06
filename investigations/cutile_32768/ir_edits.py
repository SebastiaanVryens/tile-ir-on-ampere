"""Move one feature of cuTile's Tile IR at a time towards what Triton emits, and read the SASS.

Nothing here is timed: each edit is judged by the schedule it produces. The value names are
the ones cuda-tile 1.7.0 prints for the original kernel, and every edit asserts that it
changed the text, so a different listing fails loudly instead of reporting "no effect".
"""

import re
from collections.abc import Callable

import cuda.tile as ct
import torch
from common import COLS, ROWS, SIZE
from kernels import layernorm
from tile_ir import COMPILER_HOMES, assemble, compile_bytecode, instruction_stream

from tilebench.artifacts import from_cutile, from_triton
from tilebench.cubin import Kernel, read_kernel
from tilebench.kernels.layernorm import EPS, make_inputs, triton_impl
from tilebench.schedule import read_schedule

HINTS = (3, 5)
VIEW_TYPE = re.compile(r"partition_view<tile=\((\d+(?:x\d+)?)\), ")
TENSOR_VIEW = re.compile(
    r"^(\s*)(%\w+) = make_tensor_view (%\w+), shape = \[([^\]]*)\], "
    r"strides = \[([^\]]*)\] : tile<i32> -> (.*)$",
    re.MULTILINE,
)
ZERO_INDEX = re.compile(r"%cst_0_i32\w*")

X_ROWS, X_COLS, X_STRIDE = (f"%assume_assume_assume{suffix}" for suffix in ("", "_15", "_17"))
W_COLS, B_COLS = "%assume_assume_assume_19", "%assume_assume_assume_22"
Y_ROWS, Y_COLS, Y_STRIDE = (f"%assume_assume_assume_{suffix}" for suffix in (25, 27, 29))

MASK = (
    "  %4 = iota : tile<32768xi32>\n"
    f"  %reshape_31 = reshape {X_COLS} : tile<i32> -> tile<1xi32>\n"
    "  %bcast = broadcast %reshape_31 : tile<1xi32> -> tile<32768xi32>\n"
    "  %5 = cmpi less_than %4, %bcast, signed : tile<32768xi32> -> tile<32768xi1>\n"
)
FIRST_VIEW = "  %pview = make_partition_view %tview :"
TRITON_BOUNDS = {"w": "%assume_14", "b": "%assume_17"}


def no_tokens(text: str) -> str:
    text = text.replace(" token = %0", "")
    return re.sub(r"^\s*%0 = make_token : token\n", "", text, flags=re.MULTILINE)


def strided_views(text: str) -> str:
    def view(match: re.Match[str]) -> str:
        strides = ",".join("1" * (match.group(1).count("x") + 1))
        return f"strided_view<tile=({match.group(1)}), traversal_strides=[{strides}], "

    return VIEW_TYPE.sub(view, text).replace("make_partition_view", "make_strided_view")


def wide_shapes(text: str) -> str:
    """Widen shapes and strides to i64 with the bounds Triton asserts on them."""

    def rewrite(match: re.Match[str]) -> str:
        indent, name, base, shape, strides, rest = match.groups()
        lines: list[str] = []

        def widen(value: str, bound: int, divisor: int | None = None) -> str:
            value = value.strip()
            if not value.startswith("%"):
                return value
            wide, bounded = f"{value}_{name[1:]}_w", f"{value}_{name[1:]}_b"
            lines.append(f"{indent}{wide} = exti {value} signed : tile<i32> -> tile<i64>")
            lines.append(f"{indent}{bounded} = assume bounded<0, {bound}>, {wide} : tile<i64>")
            if divisor is None:
                return bounded
            aligned = f"{value}_{name[1:]}_d"
            lines.append(f"{indent}{aligned} = assume div_by<{divisor}>, {bounded} : tile<i64>")
            return aligned

        new_shape = ", ".join(widen(value, 4294967295) for value in shape.split(","))
        new_strides = ", ".join(widen(value, 1099511627775, 8) for value in strides.split(","))
        lines.append(
            f"{indent}{name} = make_tensor_view {base}, shape = [{new_shape}], "
            f"strides = [{new_strides}] : tile<i64> -> {rest}"
        )
        return "\n".join(lines)

    return TENSOR_VIEW.sub(rewrite, text)


def aligned_index(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if "load_view_tko" in line or "store_view_tko" in line:
            head, bracket, tail = line.partition("[")
            inside, close, rest = tail.partition("]")
            for value in ZERO_INDEX.findall(inside):
                lines.append(f"  {value}_a = assume div_by<8>, {value} : tile<i32>")
                inside = inside.replace(value, f"{value}_a")
            line = head + bracket + inside + close + rest
        lines.append(line)
    return "\n".join(lines)


def target_hints(text: str) -> str:
    text = text.replace(
        "<default = {num_worker_warps_per_cta", "<sm_86 = {num_cta_in_cga = 1, num_worker_warps_per_cta"
    )
    return text.replace("<default = {latency", "<sm_86 = {latency")


def full_division(text: str) -> str:
    return re.sub(r"(divf %\w+, %\w+)\s+:", r"\1 rounding<full> :", text)


def padded_store(text: str) -> str:
    view = "partition_view<tile=(1x32768), {}tensor_view<?x?xf16, strides=[?,1]>>"
    return text.replace(view.format(""), view.format("padding_value = zero, "))


def share(pairs: tuple[tuple[str, str], ...]) -> Callable[[str], str]:
    """Point tensor views at one value, so that the compiler knows two bounds are equal."""

    def edit(text: str) -> str:
        lines = []
        for line in text.splitlines():
            if "make_tensor_view" in line:
                for old, new in pairs:
                    line = re.sub(re.escape(old) + r"(?!\w)", new, line)
            lines.append(line)
        return "\n".join(lines)

    return edit


def mask_first(text: str) -> str:
    return text.replace(MASK, "").replace(FIRST_VIEW, MASK + FIRST_VIEW)


def separate_bounds(names: tuple[str, ...]) -> Callable[[str], str]:
    """Give Triton's w and b views a runtime bound of their own, as cuTile always does."""

    def edit(text: str) -> str:
        extra = "".join(f", %N{name}: tile<i32>" for name in names)
        text = text.replace("%eps: tile<f32>)", f"%eps: tile<f32>{extra})")
        for name in names:
            old = TRITON_BOUNDS[name]
            setup = (
                f"      %n{name}_a = assume div_by<16>, %N{name} : tile<i32>\n"
                f"      %n{name}_e = exti %n{name}_a signed : tile<i32> -> tile<i64>\n"
                f"      %n{name}_b = assume bounded<0, 4294967295>, %n{name}_e : tile<i64>\n"
            )
            lines = []
            for line in text.splitlines(keepends=True):
                if "make_tensor_view" in line and f"shape = [{old}]" in line:
                    lines.append(setup)
                    line = line.replace(f"shape = [{old}]", f"shape = [%n{name}_b]")
                lines.append(line)
            text = "".join(lines)
        return text

    return edit


CUTILE_EDITS: dict[str, Callable[[str], str]] = {
    "no tokens": no_tokens,
    "strided views": strided_views,
    "i64 shapes": wide_shapes,
    "aligned index": aligned_index,
    "per-target hints": target_hints,
    "padded store": padded_store,
    "full division": full_division,
    "share y bounds": share(((Y_COLS, X_COLS), (Y_ROWS, X_ROWS))),
    "share strides": share(((X_STRIDE, X_COLS), (Y_STRIDE, X_COLS))),
    "share w, b bounds": share(((W_COLS, X_COLS), (B_COLS, X_COLS))),
    "mask first": mask_first,
}
TRITON_EDITS: dict[str, Callable[[str], str]] = {
    "separate w bound": separate_bounds(("w",)),
    "separate b bound": separate_bounds(("b",)),
    "separate w, b bounds": separate_bounds(("w", "b")),
}


def print_row(label: str, kernel: Kernel, note: str = "") -> None:
    schedule = read_schedule(kernel.sass)
    print(
        f"{label:<34}{schedule.branch_regions:>6}{schedule.first_peak:>5}"
        f"{schedule.last_peak:>6}{schedule.spills:>8}  {note}",
        flush=True,
    )


def apply_edits(
    title: str, source: str, original: Kernel, edits: dict[str, Callable[[str], str]]
) -> None:
    print(f"\n{title}")
    print(f"{'edit':<34}{'BSSY':>6}{'row':>5}{'tail':>6}{'spills':>8}")
    print_row("(none)", original)
    expected = instruction_stream(original.sass)
    for name, edit in edits.items():
        text = edit(source)
        assert text != source, f"{name} changed nothing"
        kernel = compile_bytecode(assemble(text))
        same = instruction_stream(kernel.sass) == expected
        print_row(name, kernel, "identical SASS" if same else "")


def main() -> None:
    inputs = make_inputs(SIZE)
    x, weight, bias = inputs
    y = torch.empty_like(x)
    args = (x, weight, bias, y, EPS, COLS)
    listings: dict[str, tuple[str, Kernel]] = {}
    for hint in HINTS:
        body = layernorm(hint, mask_first=False, shared_mask=False)
        built = from_cutile(ct.kernel(body, occupancy=1, num_worker_warps=8), args)
        listings[f"cutile h{hint}"] = built.stages["tileir"], read_kernel(built.cubin)
        compiled = triton_impl.whole_descriptor[(ROWS,)](
            x, weight, bias, y, ROWS, COLS, EPS, BLOCK=COLS, num_warps=8, occupancy=1, num_stages=hint
        )
        built = from_triton(compiled)
        listings[f"triton h{hint}"] = built.stages["tileir"], read_kernel(built.cubin)

    print("Round trip: does text -> bytecode -> each tileiras give back the shipped SASS?")
    for label, (text, kernel) in listings.items():
        bytecode = assemble(text)
        for compiler in COMPILER_HOMES:
            rebuilt = compile_bytecode(bytecode, compiler)
            same = instruction_stream(rebuilt.sass) == instruction_stream(kernel.sass)
            print(f"{label:<14}{compiler:<10}same SASS: {same}", flush=True)

    for hint in HINTS:
        text, kernel = listings[f"cutile h{hint}"]
        apply_edits(f"cuTile original kernel, hint {hint}", text, kernel, CUTILE_EDITS)
    for hint in HINTS:
        text, kernel = listings[f"triton h{hint}"]
        apply_edits(f"Triton descriptor kernel, hint {hint}", text, kernel, TRITON_EDITS)


if __name__ == "__main__":
    main()
