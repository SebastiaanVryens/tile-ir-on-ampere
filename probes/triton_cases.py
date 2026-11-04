"""Triton probe cases, one per process, launched by probes/run.py.

The three matmuls differ only in how they load tiles, which is the variable the
tensor-of-pointer investigation depends on. Sizes divide evenly, so no masks are needed.
"""

import json
import sys
from collections.abc import Callable
from typing import Any

import torch
import triton
import triton.language as tl

SIZE = 512
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
ADD_BLOCK = 1024


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def matmul_pointers(
    a_ptr, b_ptr, c_ptr, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
        a_ptrs += BK
        b_ptrs += BK * N
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16))


@triton.jit
def matmul_block_ptr(
    a_ptr, b_ptr, c_ptr, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    off_m = tl.program_id(0) * BM
    off_n = tl.program_id(1) * BN
    a_block = tl.make_block_ptr(a_ptr, (M, K), (K, 1), (off_m, 0), (BM, BK), (1, 0))
    b_block = tl.make_block_ptr(b_ptr, (K, N), (N, 1), (0, off_n), (BK, BN), (1, 0))
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        acc = tl.dot(tl.load(a_block), tl.load(b_block), acc)
        a_block = tl.advance(a_block, (0, BK))
        b_block = tl.advance(b_block, (BK, 0))
    c_block = tl.make_block_ptr(c_ptr, (M, N), (N, 1), (off_m, off_n), (BM, BN), (1, 0))
    tl.store(c_block, acc.to(tl.float16))


@triton.jit
def matmul_descriptor(
    a_ptr, b_ptr, c_ptr, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    off_m = tl.program_id(0) * BM
    off_n = tl.program_id(1) * BN
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BK, BN])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc = tl.dot(a_desc.load([off_m, k]), b_desc.load([k, off_n]), acc)
    c_desc.store([off_m, off_n], acc.to(tl.float16))


def scratch_alloc(size: int, alignment: int, stream: int | None) -> torch.Tensor:
    return torch.empty(size, device="cuda", dtype=torch.int8)


def summarize(compiled: Any) -> dict[str, Any]:
    """Report what actually compiled, so run.py never has to trust the environment."""
    target = compiled.metadata.target
    tile_ir = compiled.asm.get("tileir", "")
    return {
        "backend": target.backend,
        "arch": target.arch,
        "triton": triton.__version__,
        "ptr_loads": tile_ir.count("load_ptr_tko"),
        "view_loads": tile_ir.count("load_view_tko"),
    }


def run_add() -> dict[str, Any]:
    n = 1 << 20
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    compiled = add_kernel[(triton.cdiv(n, ADD_BLOCK),)](x, y, out, n, BLOCK=ADD_BLOCK)
    torch.testing.assert_close(out, x + y)
    return summarize(compiled)


def run_matmul(kernel: Any) -> dict[str, Any]:
    torch.manual_seed(0)
    a = torch.randn(SIZE, SIZE, device="cuda", dtype=torch.float16)
    b = torch.randn(SIZE, SIZE, device="cuda", dtype=torch.float16)
    c = torch.empty_like(a)
    grid = (SIZE // BLOCK_M, SIZE // BLOCK_N)
    compiled = kernel[grid](
        a, b, c, SIZE, SIZE, SIZE, BM=BLOCK_M, BN=BLOCK_N, BK=BLOCK_K
    )
    torch.testing.assert_close(c.float(), a.float() @ b.float(), rtol=2e-2, atol=1e-1)
    return summarize(compiled)


CASES: dict[str, Callable[[], dict[str, Any]]] = {
    "add": run_add,
    "matmul_pointers": lambda: run_matmul(matmul_pointers),
    "matmul_block_ptr": lambda: run_matmul(matmul_block_ptr),
    "matmul_descriptor": lambda: run_matmul(matmul_descriptor),
}


if __name__ == "__main__":
    triton.set_allocator(scratch_alloc)
    print(json.dumps(CASES[sys.argv[1]]()))
