"""Show what each compiler produced for one problem size.

Example: python -m tilebench.inspection --workload layernorm --size 512x32768 --style whole
         --where num_warps=8 num_stages=5

Prints the launch resources of every configuration, and saves its IR, SASS and cubin.
--where keeps exact matches, ignoring fields a target does not have.
"""

import argparse
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from tilebench.report import config_label, variant_name
from tilebench.targets import TARGETS, TILEIR_SITE, get_target, run_worker

# PyTorch runs precompiled kernels, so there is no build of ours to inspect.
INSPECTABLE = tuple(target.name for target in TARGETS if target.impl != "torch_impl")

OPCODE_COLUMNS = ("LDG", "LDGSTS", "LDS", "LDL", "STL", "BAR", "HMMA")

NOTE = (
    "stack is local memory per thread, which is where register spills go.\n"
    "warps/SM is the driver's estimate of how many warps fit on one SM at once.\n"
    "Opcode counts are static: a loop body counts once, however many times it runs."
)


def format_inspection(records: Iterable[dict[str, Any]]) -> str:
    opcode_header = "".join(f"{opcode:>8}" for opcode in OPCODE_COLUMNS)
    lines = [
        f"{'variant':<33}{'threads':>8}{'regs':>6}{'stack':>7}{'shared':>8}{'warps/SM':>10}"
        f"{opcode_header}{'instrs':>8}  config"
    ]
    for record in records:
        opcodes = "".join(f"{record['opcodes'][opcode]:>8}" for opcode in OPCODE_COLUMNS)
        lines.append(
            f"{variant_name(record):<33}{record['threads']:>8}{record['registers']:>6}"
            f"{record['stack_bytes']:>7}{record['shared_bytes']:>8}{record['warps_per_sm']:>10}"
            f"{opcodes}{record['instructions']:>8}  {config_label(record['config'])}"
        )
    lines.append(f"\n{NOTE}")
    return "\n".join(lines)


def inspect_targets(
    targets: Sequence[str],
    workload: str,
    size: str,
    style: str,
    where: Sequence[str],
    out_dir: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in targets:
        print(f"inspecting {workload} on {name}", flush=True)
        records += run_worker(
            get_target(name),
            workload,
            "--inspect-size",
            size,
            "--style",
            style,
            "--artifacts",
            str(out_dir),
            "--where",
            *where,
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--workload", required=True)
    parser.add_argument("--size", required=True, help="for example 512x32768")
    parser.add_argument("--style", default="", help="keep styles containing this text")
    parser.add_argument("--where", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--targets", nargs="+", default=list(INSPECTABLE))
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    if not TILEIR_SITE.is_dir():
        print(f"error: {TILEIR_SITE} not found; run scripts/setup_wsl.sh first", file=sys.stderr)
        return 2

    out_dir = args.out_dir or Path("results") / "inspect" / f"{args.workload}_{args.size}"
    records = inspect_targets(
        args.targets, args.workload, args.size, args.style, args.where, out_dir
    )
    print(format_inspection(records))
    print(f"\nIR, SASS and cubin for each configuration: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
