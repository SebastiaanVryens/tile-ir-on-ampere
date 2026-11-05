"""Triton layer norm, shared by the PTX and Tile IR targets.

The four kernels cross two strategies with two load styles. "whole" reads each row once
but must hold it all, while "looped" works in small blocks and reads each row three times.
"""

from typing import Any

import torch
import triton
import triton.language as tl

from tilebench.kernels.layernorm import (
    DESCRIPTOR_ALIGNMENT,
    EPS,
    loop_blocks,
    whole_row_block,
)
from tilebench.workload import Config, Inputs, Prepared, Size


@triton.jit
def whole_pointer(x_ptr, w_ptr, b_ptr, y_ptr, M, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    centered = tl.where(mask, x - mean, 0.0)
    rstd = 1 / tl.sqrt(tl.sum(centered * centered, axis=0) / N + eps)
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask).to(tl.float32)
    tl.store(y_ptr + row * N + cols, (centered * rstd * w + b).to(tl.float16), mask=mask)


@triton.jit
def whole_descriptor(x_ptr, w_ptr, b_ptr, y_ptr, M, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[1, BLOCK])
    y_desc = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1], block_shape=[1, BLOCK])
    w_desc = tl.make_tensor_descriptor(w_ptr, shape=[N], strides=[1], block_shape=[BLOCK])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N], strides=[1], block_shape=[BLOCK])
    mask = tl.arange(0, BLOCK) < N
    x = tl.reshape(x_desc.load([row, 0]), [BLOCK]).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    centered = tl.where(mask, x - mean, 0.0)
    rstd = 1 / tl.sqrt(tl.sum(centered * centered, axis=0) / N + eps)
    y = centered * rstd * w_desc.load([0]).to(tl.float32) + b_desc.load([0]).to(tl.float32)
    y_desc.store([row, 0], tl.reshape(y.to(tl.float16), [1, BLOCK]))


@triton.jit
def looped_pointer(x_ptr, w_ptr, b_ptr, y_ptr, M, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_ptr += row * N
    y_ptr += row * N

    total = tl.zeros([BLOCK], dtype=tl.float32)
    for offset in range(0, N, BLOCK):
        cols = offset + tl.arange(0, BLOCK)
        total += tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
    mean = tl.sum(total, axis=0) / N

    squares = tl.zeros([BLOCK], dtype=tl.float32)
    for offset in range(0, N, BLOCK):
        cols = offset + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        centered = tl.where(mask, x - mean, 0.0)
        squares += centered * centered
    rstd = 1 / tl.sqrt(tl.sum(squares, axis=0) / N + eps)

    for offset in range(0, N, BLOCK):
        cols = offset + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask).to(tl.float32)
        tl.store(y_ptr + cols, ((x - mean) * rstd * w + b).to(tl.float16), mask=mask)


@triton.jit
def looped_descriptor(x_ptr, w_ptr, b_ptr, y_ptr, M, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[1, BLOCK])
    y_desc = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1], block_shape=[1, BLOCK])
    w_desc = tl.make_tensor_descriptor(w_ptr, shape=[N], strides=[1], block_shape=[BLOCK])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N], strides=[1], block_shape=[BLOCK])

    total = tl.zeros([BLOCK], dtype=tl.float32)
    for offset in range(0, N, BLOCK):
        total += tl.reshape(x_desc.load([row, offset]), [BLOCK]).to(tl.float32)
    mean = tl.sum(total, axis=0) / N

    squares = tl.zeros([BLOCK], dtype=tl.float32)
    for offset in range(0, N, BLOCK):
        mask = offset + tl.arange(0, BLOCK) < N
        x = tl.reshape(x_desc.load([row, offset]), [BLOCK]).to(tl.float32)
        centered = tl.where(mask, x - mean, 0.0)
        squares += centered * centered
    rstd = 1 / tl.sqrt(tl.sum(squares, axis=0) / N + eps)

    for offset in range(0, N, BLOCK):
        x = tl.reshape(x_desc.load([row, offset]), [BLOCK]).to(tl.float32)
        w = w_desc.load([offset]).to(tl.float32)
        b = b_desc.load([offset]).to(tl.float32)
        y = (x - mean) * rstd * w + b
        y_desc.store([row, offset], tl.reshape(y.to(tl.float16), [1, BLOCK]))


KERNELS = {
    "whole/pointer": whole_pointer,
    "whole/descriptor": whole_descriptor,
    "looped/pointer": looped_pointer,
    "looped/descriptor": looped_descriptor,
}

# Triton's num_warps becomes the Tile IR hint num_worker_warps_per_cta, and tileiras 13.4
# stops at 8 warps, so 16 and 32 would only repeat a kernel. The occupancy hint sets the
# register budget and sizes the shared memory pipeline in the looped pointer kernels.
KNOBS = {
    "cuda": tuple({"num_warps": warps} for warps in (4, 8, 16, 32)),
    "tileir": tuple(
        {"num_warps": warps, "occupancy": occupancy}
        for warps in (4, 8)
        for occupancy in (1, 2, 4, 8)
    ),
}


def scratch_alloc(size: int, alignment: int, stream: int | None) -> torch.Tensor:
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(scratch_alloc)


def configs(backend: str, size: Size) -> tuple[Config, ...]:
    _, cols = size
    blocks = {"whole": (whole_row_block(cols),), "looped": loop_blocks(cols)}
    aligned = cols % DESCRIPTOR_ALIGNMENT == 0
    loads = ("pointer", "descriptor") if aligned else ("pointer",)
    return tuple(
        {"style": f"{strategy}/{load}", "block": block, **knob}
        for strategy, strategy_blocks in blocks.items()
        for load in loads
        for block in strategy_blocks
        for knob in KNOBS[backend]
    )


def prepare(inputs: Inputs, config: Config) -> Prepared:
    x, weight, bias = inputs
    y = torch.empty_like(x)
    rows, cols = x.shape
    kernel = KERNELS[config["style"]]
    options = {key: value for key, value in config.items() if key not in ("style", "block")}

    def launch() -> Any:
        return kernel[(rows,)](
            x, weight, bias, y, rows, cols, EPS, BLOCK=config["block"], **options
        )

    return Prepared(launch, y)
