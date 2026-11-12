"""What cuTile assumes about the GEMM arrays, and the code it builds with one more assumption.

Usage, from the activated venv: python investigations/tensor_of_pointer/signature.py

When cuTile compiles a kernel just in time, it derives a signature from the arguments: for
each array, what its base address, strides and shape are divisible by. This prints that
signature at 4,096 and 4,088, then compiles the GEMM ahead of time with three signatures
and compares the SASS:
  4,096 and 4,088, as the JIT derives them
  4,088 with the shapes declared divisible by 8, which is true
tilebench.artifacts.from_cutile exports the same way, and with this calling convention the
exported cubin is the one ct.launch runs. The builds are not timed: launching an exported
kernel would need a hand-written launch against cuTile's calling convention.
"""

import io
from typing import Any

import torch
from common import describe_loads, instruction_count, instructions, load_forms, main
from cuda.tile.compilation import (
    ArrayConstraint,
    CallingConvention,
    KernelSignature,
    export_kernel,
)

from tilebench.cubin import read_kernel
from tilebench.kernels.gemm import GROUP_ROWS, TUTORIAL_TILES, cutile_impl, make_inputs

TILES = ((128, 128, 32), (128, 256, 64))
# The setting cuTile picked at 4,088 on 128x128x32.
OCCUPANCY, LATENCY = 1, 1
DECLARED = "4,088, shapes divisible by 8"


def run(name: str) -> dict[str, Any]:
    convention = CallingConvention.cutile_python_v1()
    derived: dict[str, Any] = {}
    builds = []
    for tile in TILES:
        warps = cutile_impl.worker_warps(next(w for m, n, k, w in TUTORIAL_TILES if (m, n, k) == tile))
        kernel = cutile_impl.build_kernel(OCCUPANCY, LATENCY, warps)
        signatures = {}
        for size in (4096, 4088):
            a, b = make_inputs((size, size, size))
            c = torch.empty(size, size, device=a.device, dtype=a.dtype)
            args = (a, b, c, *tile, GROUP_ROWS)
            signatures[f"{size:,}"] = KernelSignature.from_kernel_args(kernel, args, convention)
        signatures[DECLARED] = with_shapes_divisible_by(signatures["4,088"], 8)
        for label, signature in signatures.items():
            derived[label] = [
                {"shape": p.shape_divisible_by, "stride": p.stride_divisible_by,
                 "base": p.base_addr_divisible_by}
                for p in signature.parameters[:3]
            ]
            build = read_kernel(export(kernel, signature))
            builds.append({"tile": tile, "signature": label, "registers": build.registers,
                           "stack_bytes": build.stack_bytes,
                           "instructions": instruction_count(build.sass),
                           "loads": load_forms(build.sass), "code": instructions(build.sass)})
    return {"derived": derived, "builds": builds}


def with_shapes_divisible_by(signature: KernelSignature, divisor: int) -> KernelSignature:
    """Return the signature with every array's shape declared divisible by `divisor`."""
    parameters = []
    for p in signature.parameters:
        if isinstance(p, ArrayConstraint):
            p = ArrayConstraint(
                p.dtype,
                len(p.shape_divisible_by),
                index_dtype=p.index_dtype,
                stride_lower_bound_incl=p.stride_lower_bound_incl,
                alias_groups=p.alias_groups,
                may_alias_internally=p.may_alias_internally,
                stride_constant=p.stride_constant,
                shape_constant=p.shape_constant,
                stride_divisible_by=p.stride_divisible_by,
                shape_divisible_by=divisor,
                base_addr_divisible_by=p.base_addr_divisible_by,
            )
        parameters.append(p)
    return KernelSignature(parameters, signature.calling_convention).with_mangled_symbol("gemm")


def export(kernel: Any, signature: KernelSignature) -> bytes:
    major, minor = torch.cuda.get_device_capability()
    buffer = io.BytesIO()
    export_kernel(kernel, [signature], buffer, gpu_code=f"sm_{major}{minor}", output_format="cubin")
    return buffer.getvalue()


def report(results: dict[str, Any]) -> None:
    result = results["cutile"]
    print("\nWhat each array is divisible by in the signature: shape, strides, base address in bytes")
    for label, arrays in result["derived"].items():
        cells = "   ".join(f"{name}: {tuple(p['shape'])}, {tuple(p['stride'])}, {p['base']}"
                           for name, p in zip("abc", arrays, strict=True))
        print(f"  {label:<30}{cells}")

    print(f"\nAhead-of-time builds, occupancy {OCCUPANCY}, latency {LATENCY}")
    print(f"  {'tile':<12}{'signature':<31}{'regs':>5}{'stack':>7}{'instrs':>8}   loads")
    for build in result["builds"]:
        tile = "x".join(map(str, build["tile"]))
        print(f"  {tile:<12}{build['signature']:<31}{build['registers']:>5}{build['stack_bytes']:>7}"
              f"{build['instructions']:>8}   {describe_loads(build['loads'])}")
    print()
    for tile in TILES:
        code = {b["signature"]: b["code"] for b in result["builds"] if tuple(b["tile"]) == tile}
        for label in ("4,088", DECLARED):
            verdict = "the same SASS, line for line" if code[label] == code["4,096"] else "different SASS"
            print(f"  {'x'.join(map(str, tile))}, {label} against 4,096: {verdict}")


if __name__ == "__main__":
    main(__file__, ("cutile",), run, report)
