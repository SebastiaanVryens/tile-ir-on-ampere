"""Three ways to tell Triton that the sizes are multiples of 8, and what each one changes.

Usage, from the activated venv: python investigations/tensor_of_pointer/hints.py

At 4,088 on both backends, both kernels are built plain and with each of these, at the
reference setting (128x256x64, 8 warps, 3 stages or hint 3, occupancy 1 on Tile IR):
  hint     tl.multiple_of(K, 8), the language's divisibility hint
  assume   tl.assume(K % 8 == 0)
  round    K // 8 * 8 passed in place of K
each applied to M, N and K. Nothing is timed. The table counts the divisibility attributes
and assumptions in the Triton IR, describes the build, and says whether the SASS is the
same as the plain kernel's.
"""

from typing import Any

from common import (
    REFERENCE_KNOBS,
    REFERENCE_TILE,
    compiled,
    describe_loads,
    instruction_count,
    instructions,
    load_forms,
    main,
)

from tilebench.kernels.gemm import tile_fields
from tilebench.targets import get_target
from tilebench.worker import load_impl, load_workload

TARGETS = ("triton-ptx", "triton-tileir")
SIZE = (4088, 4088, 4088)
WAYS = ("", "hint", "assume", "round")


def run(name: str) -> list[dict[str, Any]]:
    import kernels

    kernels.register()
    target = get_target(name)
    workload = load_workload("gemm")
    impl = load_impl("gemm", target)
    rows = []
    for style in ("pointer", "descriptor"):
        for way in WAYS:
            config = {"style": f"{style}-{way}" if way else style, **tile_fields(*REFERENCE_TILE),
                      **REFERENCE_KNOBS[target.backend]}
            artifacts, kernel = compiled(target, workload, impl, SIZE, config)
            ttir = artifacts.stages["ttir"]
            rows.append({"style": style, "way": way,
                         "divisibility": ttir.count("tt.divisibility"),
                         "assumes": ttir.count("llvm.intr.assume"),
                         "registers": kernel.registers, "stack_bytes": kernel.stack_bytes,
                         "instructions": instruction_count(kernel.sass),
                         "loads": load_forms(kernel.sass), "code": instructions(kernel.sass)})
    return rows


def report(results: dict[str, Any]) -> None:
    print(f"\n{SIZE[0]:,}, 128x256x64. 'div.' counts tt.divisibility attributes in the Triton IR, "
          "'assume' counts llvm.intr.assume.")
    print(f"{'backend':<15}{'kernel':<20}{'div.':>5}{'assume':>8}{'regs':>6}{'stack':>7}{'instrs':>8}"
          f"   {'same SASS as plain':<20}loads")
    for name, rows in results.items():
        plain = {row["style"]: row["code"] for row in rows if not row["way"]}
        for row in rows:
            kernel = f"{row['style']}-{row['way']}" if row["way"] else row["style"]
            same = "-" if not row["way"] else ("yes" if row["code"] == plain[row["style"]] else "no")
            print(f"{name:<15}{kernel:<20}{row['divisibility']:>5}{row['assumes']:>8}{row['registers']:>6}"
                  f"{row['stack_bytes']:>7}{row['instructions']:>8}   {same:<20}{describe_loads(row['loads'])}")


if __name__ == "__main__":
    main(__file__, TARGETS, run, report)
