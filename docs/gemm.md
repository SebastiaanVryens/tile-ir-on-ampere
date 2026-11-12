# GEMM results

The GEMM multiplies two square fp16 matrices with fp32 accumulation, from 1,024 to 8,192 on a side. It is the first compute-bound kernel here: the tensor cores set the limit, not memory bandwidth.

Every path tries the same 11 tile shapes, the Ampere configurations of Triton's matrix multiplication tutorial, and visits the output tiles in bands of 8 block rows so that blocks running together share their inputs in L2. On top of that:

- Triton tries pointer and descriptor loads, with the tutorial's warp count for each tile. On PTX, tuning sweeps 2 to 5 pipeline stages where they fit in 99 KB of shared memory. On Tile IR, it sweeps the latency hint (1, 3, 5, 10) and occupancy (1, 2).
- cuTile sweeps the same hints, plus no hint, and the same occupancies. The compiler picks the warp count, except on the largest tile, as described below.
- PyTorch calls cuBLAS through `torch.mm`. By default, PyTorch lets cuBLAS reduce partial sums in fp16, which put 6% of the results at 256×256×8,192 outside the tolerance, so that is turned off. Doing so moved speed at the benchmark sizes by at most 0.5%.

That makes 371 configurations per size. Each one is checked against an fp32 matrix multiply rounded to fp16, including at awkward sizes such as 65×129×1,025.

TFLOPS for the best configuration of each path and size:

| Size | PyTorch (cuBLAS) | cuTile | Triton PTX, pointer | Triton PTX, descriptor | Triton Tile IR, pointer | Triton Tile IR, descriptor |
|---|---|---|---|---|---|---|
| 1,024 | 14.7 | 22.5 | 23.8 | 23.8 | 22.6 | 21.8 |
| 2,048 | 26.9 | 26.2 | 27.1 | 26.5 | 26.1 | 25.5 |
| 4,096 | 27.2 | 26.6 | 27.1 | 26.2 | 26.4 | 25.7 |
| 8,192 | 27.4 | 26.7 | 27.2 | 26.3 | 27.0 | 25.2 |

- From 2,048 up, every path is within 8% of cuBLAS. At 8,192, Triton PTX and Triton Tile IR with pointer loads reach 99% of it, and cuTile reaches 97%.
- At 1,024, cuBLAS is the slowest path by far, for the reason explained below. The Tile IR README says: "Small GEMM performance is currently poor." Here the Tile IR paths take 6 to 9% longer than Triton PTX at 1,024.
- Pointer loads beat descriptor loads on both Triton backends from 2,048 up. On Tile IR, the best descriptor kernel takes 2.5%, 2.9% and 7.3% longer than the best pointer kernel. Comparing the same tile during tuning, pointer loads were faster on 10 of the 11 tiles at 4,096 and on 9 of 11 at 8,192. The Tile IR README says that kernels using the "legacy tensor-of-pointer load/store APIs exhibit poor performance". On this GPU it is the other way around, which is where the tensor-of-pointer investigation starts.

## Why cuBLAS is slowest at 1,024

The PyTorch profiler shows which kernel cuBLAS runs. At 1,024 it is `ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_32x3_nn`, which computes a 256×128 output tile per block. Its registers (218 per thread over 8 warps) and shared memory (72 KB) allow one block per SM at a time.

A 1,024×1,024 output has 32 such tiles, and the GPU has 30 SMs. So 30 blocks run, and then the last 2 run alone while 28 SMs sit idle. The kernel takes two rounds for 1.07 rounds of work, which uses about 53% of the GPU, and 14.7 TFLOPS is 54% of the 27.4 that cuBLAS reaches at 8,192. The tuned tile kernels use 128×32 tiles at this size, which gives 256 smaller blocks.

At 4,096, cuBLAS uses the same tile shape in 512 blocks, so the idle part of the last round costs at most about 5%. At 8,192 it switches to a 128×128 tile.

## cuTile's warp count on the largest tile

cuTile picks the warp count itself. On the largest tile, 128×256×64, it picked 4 warps. The fp32 accumulator alone then needs 256 registers per thread, more than the limit of 255, so the kernel spilled 1,112 bytes per thread to local memory and ran at 6.6 TFLOPS at 4,096. With `num_worker_warps=8` it used 222 registers, did not spill, and ran at 22.9 TFLOPS, 3.5 times faster.

The Triton paths use the tutorial's 8 warps on that tile, so cuTile gets 8 there as well. Every other tile keeps the compiler's choice, which was 4 warps on every tile inspected. The cuTile documentation describes `num_worker_warps` as a setting for warp-specialized kernels, but on sm_86 it set the block size directly. The tile never won for cuTile and stayed 9% behind its best at 8,192, so the change made the comparison fair without moving cuTile's results.

## Occupancy, pipeline depth and the latency hint

Occupancy 2 won for most Tile IR paths and sizes. Because 2 is the top of the sweep, occupancy 3 and 4 were tried at 4,096 on every tile within 4% of the best, and they never beat the best result at occupancy 2. To fit more blocks per SM, the compiler caps the registers per thread, at 168 for occupancy 3 and 128 for occupancy 4 with 128 threads, and it shrinks the shared-memory pipeline. Tiles whose accumulator no longer fits then spill. cuTile in TFLOPS, best over hints at 4,096:

| Tile | Occupancy 2 | Occupancy 3 | Occupancy 4 |
|---|---|---|---|
| 64×128×32 | 25.1 (160 regs, 48 KB) | 25.5 (146 regs, 28 KB) | 23.6 (128 regs, 20 KB, spills 40 B) |
| 128×128×32 | 25.8 (228 regs) | 20.2 (168 regs, spills 176 B) | 2.7 (128 regs, spills 504 B) |
| 64×256×32 | 26.7 (216 regs) | 17.8 (168 regs, spills 192 B) | 23.7 (one block per SM) |

Sometimes the compiler does not meet the request at all. 64×256×32 at occupancy 4 got one block per SM, and 128×256×64 builds the same at occupancy 1 to 4, also with one block per SM. Triton Tile IR behaves the same way, since both paths compile through tileiras.

On PTX, the number of pipeline stages is the matching trade-off, and 2 or 3 stages won. One stage, tried at 8,192 on the five closest tiles, was 16 to 29% slower: Triton then emits no asynchronous copies (`LDGSTS`), so every step along K waits for its loads. More stages cost shared memory. On 64×256×32, a fourth stage takes it from 40 to 60 KB, only one block fits per SM instead of two, and speed drops by about 10%.

The latency hint mattered less than for [layernorm](layernorm.md#the-latency-hint). On Triton Tile IR it moved each best result by at most 4%. On cuTile it mattered at 8,192, where the best result was 24.6 TFLOPS without a hint and 26.8 with a hint of 1.

## What the compilers produced

On the same tile, 64×256×32 at occupancy 2, cuTile and both Triton Tile IR kernels build the same structure at 8,192: 128 threads, 36 KB of shared memory, two blocks per SM, 128 `HMMA.16816.F32` tensor core instructions, 20 asynchronous copies and 32 `LDSM` matrix loads from shared memory. So Tile IR turns tensor-of-pointer loads into the same asynchronous copies as tile loads.

Two things differ. The pointer kernel's copies skip L1 (`LDGSTS.E.BYPASS.128`), and those of the descriptor and cuTile kernels do not (`LDGSTS.E.128`). Triton PTX, inspected on 128×256×64, uses the bypassing form for both load styles. The bounds checks differ as well: per step along K, the main loop of the pointer kernel has 187 instructions with 1 comparison, against 208 with 29 for the descriptor kernel and 199 with 29 for cuTile.

During tuning on this tile, cuTile reached 26.8 TFLOPS, Triton Tile IR 26.1 with pointer loads and 25.4 with descriptors. cuTile is the fastest of the three with neither the copies that skip L1 nor the shortest loop, so neither difference explains the speeds on its own. On 128×256×64, pointer loads are 6% faster, and the descriptor kernel's loop has 315 instructions per step with 34 comparisons, against 270 with 1. [The tensor-of-pointer investigation](../investigations/tensor_of_pointer/README.md) compares the main loops on both backends.

One result is not yet explained. At 8,192, Triton PTX kernels with narrow tiles slowed down sharply, with both load styles and every stage count: 128×32×32 ran at 4.8 TFLOPS against 21.1 at 4,096, and 128×64×32 at about 10 against 25. The samples were steady and the clocks normal, so it is not a measurement problem, and Triton Tile IR slowed far less on the same tiles. These tiles never win at that size.
