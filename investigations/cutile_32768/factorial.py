"""Which changes close the gap? Every combination of the two fixes and full division, at every hint."""

import cuda.tile as ct
import torch
from common import COLS, HINTS, ROWS, SIZE, measure, print_by_hint, print_kernels, verified
from kernels import layernorm

from tilebench.artifacts import from_cutile
from tilebench.cubin import read_kernel
from tilebench.kernels.layernorm import EPS, make_inputs, reference

FIXES = {
    "original": (False, False),
    "mask first": (True, False),
    "shared mask": (False, True),
    "both": (True, True),
}
DIVISIONS = {"": None, " / full": ct.RoundingMode.FULL}
SHOWN_HINTS = (3, 5)


def main() -> None:
    inputs = make_inputs(SIZE)
    x, weight, bias = inputs
    expected = reference(*inputs)
    y = torch.empty_like(x)
    args = (x, weight, bias, y, EPS, COLS)
    launches, kernels = {}, {}
    for fix, (mask_first, shared_mask) in FIXES.items():
        for suffix, rounding in DIVISIONS.items():
            for hint in HINTS:
                label = f"{fix}{suffix} h{hint}"
                body = layernorm(hint, mask_first, shared_mask, rounding)
                kernel = ct.kernel(body, occupancy=1, num_worker_warps=8)

                def launch(kernel: object = kernel) -> None:
                    ct.launch(torch.cuda.current_stream(), (ROWS, 1, 1), kernel, args)

                verified(launch, y, expected)
                launches[label] = launch
                if hint in SHOWN_HINTS:
                    kernels[label] = read_kernel(from_cutile(kernel, args).cubin)

    timings = measure(launches)
    print_by_hint(tuple(f"{fix}{suffix}" for suffix in DIVISIONS for fix in FIXES), timings)
    print()
    print_kernels(kernels, timings)


if __name__ == "__main__":
    main()
