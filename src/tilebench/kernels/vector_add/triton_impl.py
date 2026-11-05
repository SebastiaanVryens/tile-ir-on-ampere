"""Triton vector add, shared by the PTX and Tile IR targets."""

from typing import Any

import torch
import triton
import triton.language as tl

from tilebench.workload import Config, Inputs, Prepared, Size


@triton.jit
def add_pointers(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def add_descriptor(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[n], strides=[1], block_shape=[BLOCK])
    y_desc = tl.make_tensor_descriptor(y_ptr, shape=[n], strides=[1], block_shape=[BLOCK])
    out_desc = tl.make_tensor_descriptor(out_ptr, shape=[n], strides=[1], block_shape=[BLOCK])
    out_desc.store([offset], x_desc.load([offset]) + y_desc.load([offset]))


KERNELS = {"pointer": add_pointers, "descriptor": add_descriptor}

# occupancy stays at its default: sweeping it from 1 to 8 changed nothing measurable.
BLOCKS_AND_WARPS = ((1024, 4), (4096, 8))


def scratch_alloc(size: int, alignment: int, stream: int | None) -> torch.Tensor:
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(scratch_alloc)


def configs(backend: str, size: Size) -> tuple[Config, ...]:
    return tuple(
        {"style": style, "block": block, "num_warps": num_warps}
        for style in KERNELS
        for block, num_warps in BLOCKS_AND_WARPS
    )


def prepare(inputs: Inputs, config: Config) -> Prepared:
    x, y = inputs
    out = torch.empty_like(x)
    n = x.numel()
    block = config["block"]
    kernel = KERNELS[config["style"]]
    grid = (triton.cdiv(n, block),)

    def launch() -> Any:
        return kernel[grid](x, y, out, n, BLOCK=block, num_warps=config["num_warps"])

    return Prepared(launch, out)
