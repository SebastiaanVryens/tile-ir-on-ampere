"""Triton GEMM variants next to the repository's two kernels.

Each variant changes either how the tiles are loaded or what the compiler knows about the
sizes, and nothing else. `register` adds them to the benchmark's kernel table under their
style names, so the benchmark's configurations, checks and timing apply to them unchanged.
Only import this module in a Triton target's process.
"""

import triton
import triton.language as tl

from tilebench.kernels.gemm import triton_impl
from tilebench.kernels.gemm.triton_impl import block_position, gemm_descriptor, gemm_pointer


@triton.jit
def gemm_pointer_masked(
    a_ptr, b_ptr, c_ptr, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
):
    """Pointer loads written the plain way: masked along both dimensions, with no wrapping."""
    block_m, block_n = block_position(M, N, BM, BN, GROUP)
    rows = block_m * BM + tl.arange(0, BM)
    cols = block_n * BN + tl.arange(0, BN)
    steps = tl.arange(0, BK)
    a_ptrs = a_ptr + rows[:, None] * K + steps[None, :]
    b_ptrs = b_ptr + steps[:, None] * N + cols[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs, mask=(rows[:, None] < M) & (steps[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(steps[:, None] < K - k) & (cols[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK * N
    c_ptrs = c_ptr + rows[:, None] * N + cols[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=(rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def gemm_pointer_ca(
    a_ptr, b_ptr, c_ptr, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
):
    """The repository's pointer kernel, with loads that ask to be cached in L1."""
    block_m, block_n = block_position(M, N, BM, BN, GROUP)
    rows = block_m * BM + tl.arange(0, BM)
    cols = block_n * BN + tl.arange(0, BN)
    steps = tl.arange(0, BK)
    a_ptrs = a_ptr + (rows % M)[:, None] * K + steps[None, :]
    b_ptrs = b_ptr + steps[:, None] * N + (cols % N)[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs, mask=steps[None, :] < K - k, other=0.0, cache_modifier=".ca")
        b = tl.load(b_ptrs, mask=steps[:, None] < K - k, other=0.0, cache_modifier=".ca")
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK * N
    c_ptrs = c_ptr + rows[:, None] * N + cols[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=(rows[:, None] < M) & (cols[None, :] < N))


# Three ways to tell the compiler that M, N and K are multiples of 8. Each one wraps an
# unchanged kernel of the repository, so the kernel bodies stay the same.


@triton.jit
def gemm_pointer_hint(
    a_ptr, b_ptr, c_ptr, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
):
    """The language's divisibility hint, on the size arguments."""
    M = tl.multiple_of(M, 8)
    N = tl.multiple_of(N, 8)
    K = tl.multiple_of(K, 8)
    gemm_pointer(a_ptr, b_ptr, c_ptr, M, N, K, BM, BN, BK, GROUP)


@triton.jit
def gemm_descriptor_hint(
    a_ptr, b_ptr, c_ptr, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
):
    M = tl.multiple_of(M, 8)
    N = tl.multiple_of(N, 8)
    K = tl.multiple_of(K, 8)
    gemm_descriptor(a_ptr, b_ptr, c_ptr, M, N, K, BM, BN, BK, GROUP)


@triton.jit
def gemm_pointer_assume(
    a_ptr, b_ptr, c_ptr, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
):
    """An assumption the compiler may use."""
    tl.assume(M % 8 == 0)
    tl.assume(N % 8 == 0)
    tl.assume(K % 8 == 0)
    gemm_pointer(a_ptr, b_ptr, c_ptr, M, N, K, BM, BN, BK, GROUP)


@triton.jit
def gemm_descriptor_assume(
    a_ptr, b_ptr, c_ptr, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
):
    tl.assume(M % 8 == 0)
    tl.assume(N % 8 == 0)
    tl.assume(K % 8 == 0)
    gemm_descriptor(a_ptr, b_ptr, c_ptr, M, N, K, BM, BN, BK, GROUP)


@triton.jit
def gemm_pointer_round(
    a_ptr, b_ptr, c_ptr, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
):
    """Sizes rounded down to a multiple of 8, which leaves them unchanged when they are one.

    The compiler can then derive the divisibility from the arithmetic. The caller must
    check that the sizes are multiples of 8, or the kernel computes on smaller sizes.
    """
    gemm_pointer(a_ptr, b_ptr, c_ptr, M // 8 * 8, N // 8 * 8, K // 8 * 8, BM, BN, BK, GROUP)


@triton.jit
def gemm_descriptor_round(
    a_ptr, b_ptr, c_ptr, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
):
    gemm_descriptor(a_ptr, b_ptr, c_ptr, M // 8 * 8, N // 8 * 8, K // 8 * 8, BM, BN, BK, GROUP)


# Triton marks integer arguments divisible by 16 when it compiles a kernel. These copies of
# the repository's kernels do without that, so a size of 4,096 looks like any other.
SIZE_ARGUMENTS = ["M", "N", "K"]

VARIANTS = {
    "pointer-masked": gemm_pointer_masked,
    "pointer-ca": gemm_pointer_ca,
    "pointer-hint": gemm_pointer_hint,
    "descriptor-hint": gemm_descriptor_hint,
    "pointer-assume": gemm_pointer_assume,
    "descriptor-assume": gemm_descriptor_assume,
    "pointer-round": gemm_pointer_round,
    "descriptor-round": gemm_descriptor_round,
    "pointer-nospec": triton.jit(do_not_specialize=SIZE_ARGUMENTS)(gemm_pointer.fn),
    "descriptor-nospec": triton.jit(do_not_specialize=SIZE_ARGUMENTS)(gemm_descriptor.fn),
}


def register() -> None:
    triton_impl.KERNELS.update(VARIANTS)
