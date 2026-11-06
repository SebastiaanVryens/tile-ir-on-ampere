"""The test that ruled out division: give Triton IEEE division and see whether it slows down.

Triton's / lowers to approximate division, and cuTile's to IEEE division. If that were part
of the gap, these kernels would lose speed. They do not.
"""

import torch
import triton
import triton.language as tl
from common import COLS, HINTS, ROWS, SIZE, measure, print_by_hint, print_kernels, verified

from tilebench.artifacts import from_triton
from tilebench.cubin import read_kernel
from tilebench.kernels.layernorm import EPS, make_inputs, reference, triton_impl

SHOWN_HINTS = (3, 5)


@triton.jit
def pointer_ieee(x_ptr, w_ptr, b_ptr, y_ptr, M, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    count = N.to(tl.float32)
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.div_rn(tl.sum(x, axis=0), count)
    centered = tl.where(mask, x - mean, 0.0)
    rstd = tl.div_rn(1.0, tl.sqrt(tl.div_rn(tl.sum(centered * centered, axis=0), count) + eps))
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask).to(tl.float32)
    tl.store(y_ptr + row * N + cols, (centered * rstd * w + b).to(tl.float16), mask=mask)


@triton.jit
def descriptor_ieee(x_ptr, w_ptr, b_ptr, y_ptr, M, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[1, BLOCK])
    y_desc = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1], block_shape=[1, BLOCK])
    w_desc = tl.make_tensor_descriptor(w_ptr, shape=[N], strides=[1], block_shape=[BLOCK])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N], strides=[1], block_shape=[BLOCK])
    mask = tl.arange(0, BLOCK) < N
    count = N.to(tl.float32)
    x = tl.reshape(x_desc.load([row, 0]), [BLOCK]).to(tl.float32)
    mean = tl.div_rn(tl.sum(x, axis=0), count)
    centered = tl.where(mask, x - mean, 0.0)
    rstd = tl.div_rn(1.0, tl.sqrt(tl.div_rn(tl.sum(centered * centered, axis=0), count) + eps))
    y = centered * rstd * w_desc.load([0]).to(tl.float32) + b_desc.load([0]).to(tl.float32)
    y_desc.store([row, 0], tl.reshape(y.to(tl.float16), [1, BLOCK]))


VARIANTS = {
    "pointer": triton_impl.whole_pointer,
    "pointer ieee": pointer_ieee,
    "descriptor": triton_impl.whole_descriptor,
    "descriptor ieee": descriptor_ieee,
}


def main() -> None:
    inputs = make_inputs(SIZE)
    x, weight, bias = inputs
    expected = reference(*inputs)
    y = torch.empty_like(x)
    launches, kernels = {}, {}
    for name, kernel in VARIANTS.items():
        for hint in HINTS:
            label = f"{name} h{hint}"

            def launch(kernel: triton.JITFunction = kernel, hint: int = hint) -> object:
                return kernel[(ROWS,)](
                    x, weight, bias, y, ROWS, COLS, EPS,
                    BLOCK=COLS, num_warps=8, occupancy=1, num_stages=hint,
                )  # fmt: skip

            compiled = verified(launch, y, expected)
            launches[label] = launch
            if hint in SHOWN_HINTS:
                kernels[label] = read_kernel(from_triton(compiled).cubin)

    timings = measure(launches)
    print_by_hint(tuple(VARIANTS), timings)
    print()
    print_kernels(kernels, timings)


if __name__ == "__main__":
    main()
