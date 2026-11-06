"""Does the mask position matter on other GPUs? Compile only, so no such GPU is needed.

This kernel is cut down to the row load and one use of the mask. The two versions differ in
one line, and each cell is the number of branch regions in the compiled code.
"""

import io
from collections.abc import Callable

import cuda.tile as ct
import torch
from common import COLS, SIZE
from cuda.tile.compilation import CallingConvention, KernelSignature, export_kernel

from tilebench.cubin import count_opcodes, read_kernel
from tilebench.kernels.layernorm import make_inputs

ZERO = ct.PaddingMode.ZERO
TARGETS = ("sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120")
HINTS = (3, 5)


def mask_after(latency: int) -> Callable[..., None]:
    def whole(x, y, tile: ct.Constant[int]):
        bid = ct.bid(0)
        n = x.shape[1]
        row = ct.load(x, index=(bid, 0), shape=(1, tile), padding_mode=ZERO, latency=latency)
        row = ct.astype(row, ct.float32)
        mean = ct.sum(row, axis=1) / n
        mask = ct.arange(tile, dtype=ct.int32) < n
        centered = ct.where(mask, row - mean, 0)
        ct.store(y, index=(bid, 0), tile=ct.astype(centered, y.dtype), latency=latency)

    return whole


def mask_before(latency: int) -> Callable[..., None]:
    def whole(x, y, tile: ct.Constant[int]):
        bid = ct.bid(0)
        n = x.shape[1]
        mask = ct.arange(tile, dtype=ct.int32) < n
        row = ct.load(x, index=(bid, 0), shape=(1, tile), padding_mode=ZERO, latency=latency)
        row = ct.astype(row, ct.float32)
        mean = ct.sum(row, axis=1) / n
        centered = ct.where(mask, row - mean, 0)
        ct.store(y, index=(bid, 0), tile=ct.astype(centered, y.dtype), latency=latency)

    return whole


def branch_regions(body: Callable[..., None], args: tuple[object, ...], target: str) -> int:
    kernel = ct.kernel(body, occupancy=1, num_worker_warps=8)
    signature = KernelSignature.from_kernel_args(kernel, args, CallingConvention.cutile_python_v1())
    buffer = io.BytesIO()
    export_kernel(kernel, [signature], buffer, gpu_code=target, output_format="cubin")
    return count_opcodes(read_kernel(buffer.getvalue()).sass)["BSSY"]


def main() -> None:
    x = make_inputs(SIZE)[0]
    args = (x, torch.empty_like(x), COLS)
    labels = [f"h{hint} {place}" for hint in HINTS for place in ("after", "before")]
    print(f"{'target':<9}" + "".join(f"{label:>11}" for label in labels))
    for target in TARGETS:
        cells = [
            branch_regions(build(hint), args, target)
            for hint in HINTS
            for build in (mask_after, mask_before)
        ]
        print(f"{target:<9}" + "".join(f"{cell:>11}" for cell in cells), flush=True)


if __name__ == "__main__":
    main()
