# Tile IR on Ampere: cuTile and Triton compared

In this project, I will write the same GPU kernels in three ways: cuTile Python, Triton on the PTX backend, and Triton on the CUDA Tile IR backend. I used PyTorch as the reference for correctness and performance.
Why, you would ask? The goal is to understand how each path compiles and how kernels actually work.

## Status

- [x] Pinned environment
- [x] Probe: every path compiles and runs correctly on sm_86
- [x] Benchmark harness
- [x] Kernel: vector add
- [x] Kernel: layernorm
- [x] Investigation: why cuTile trailed at 32,768 columns
- [x] Kernel: tiled GEMM
- [ ] Tensor-of-pointer investigation

Vector add validates the harness. Layernorm is the first kernel where the paths differ, at 65,536 columns. GEMM is the first compute-bound kernel, and from 2,048 up every path comes within 8% of cuBLAS.

## Test setup

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU (sm_86, 6 GB) |
| Driver | 610.88 |
| OS | Windows 11, WSL2 Ubuntu 24.04 |
| Python | 3.12 |
| PyTorch | 2.15.0 |
| cuTile | cuda-tile 1.7.0 with tileiras 13.4.92 |
| Triton (upstream) | 3.9.0 |
| Triton (Tile IR wheel) | triton_tileir 3.7.1 with tileiras 13.4.59 |

## Setup and run

You need WSL2 Ubuntu 24.04 with the GPU visible to `nvidia-smi`, and about 15 GB free on the Windows drive. From the repository root:

```bash
bash scripts/setup_wsl.sh
source ~/.venvs/cutile/bin/activate
python probes/run.py
pytest
python -m tilebench
```

`python -m tilebench` prints a comparison table and saves the raw timings in `results/`. A full run takes about two hours, most of it GEMM tuning, which waits for the GPU to cool between bursts. The first run takes longer while every kernel compiles. `--workloads` runs a subset, for example `python -m tilebench --workloads gemm`.

## Probe results

The probe checks correctness only. Each case runs in its own process and is compared against PyTorch:

- `add` is a vector add over 2^20 fp32 elements.
- The `matmul` cases are 512×512×512 fp16 matrix multiplies with fp32 accumulation, and differ only in how they load tiles.

A case also fails if the compiled kernel did not use the expected backend. `triton-ptx` and `triton-tileir` come from the same wheel, so they differ only in the backend.

```text
target          case                result backend  triton   ptr / view loads
triton-upstream add                 PASS   cuda     3.9.0    -
triton-upstream matmul_pointers     PASS   cuda     3.9.0    -
triton-upstream matmul_block_ptr    fail*  -        -        -
triton-upstream matmul_descriptor   PASS   cuda     3.9.0    -
triton-ptx      add                 PASS   cuda     3.7.1    -
triton-ptx      matmul_pointers     PASS   cuda     3.7.1    -
triton-ptx      matmul_block_ptr    PASS   cuda     3.7.1    -
triton-ptx      matmul_descriptor   PASS   cuda     3.7.1    -
triton-tileir   add                 PASS   tileir   3.7.1    2 / 0
triton-tileir   matmul_pointers     PASS   tileir   3.7.1    2 / 0
triton-tileir   matmul_block_ptr    PASS   tileir   3.7.1    2 / 0
triton-tileir   matmul_descriptor   PASS   tileir   3.7.1    0 / 2
cutile          add                 PASS   cutile   -        -
cutile          matmul              PASS   cutile   -        -
```

`fail*` marks an optional case: Triton 3.9 removed block pointers in favor of tensor descriptors, so that kernel no longer compiles there.

`ptr / view loads` counts the `load_ptr_tko` and `load_view_tko` operations in the generated Tile IR.

The Triton Tile IR release notes list validation on B200. These small cases also ran correctly on sm_86, which is a much narrower check than that validation.

## How it is measured

`python -m tilebench` has two stages. Tuning times every configuration briefly and keeps the fastest one for each variant and size. Three rounds then measure only those winners, so the reported numbers do not come from the timings that picked them. Each reported value is the median of 300 launches.

Every configuration is checked against PyTorch before it is timed, and the L2 cache is cleared before each launch. A laptop GPU needs two more precautions:

- **Continuous load.** Clocks only stay up under steady load. Timing one kernel at a time with about 100 ms gaps ran near 580 MHz and read up to 2.5x too slow. So kernels are compiled and checked first, then timed in bursts: one 0.75 s warmup, then launches back to back.
- **A cool start.** The median SM clock holds at 1950 MHz up to 82 C, then falls to 1575 MHz at 86 C. Each burst waits until the GPU is at 70 C or below.
- **Shorter, cooler bursts for GEMM.** A burst is normally 640 launches, which at 8,192 would be 26 s of GEMM. Under cuBLAS at that size, the GPU reaches its 130 W power limit within a second, warms by about 1 C per second, and slows for heat at 86 to 87 C after about 30 s. So a burst also ends after 5 s, and GEMM bursts wait for 60 C. In a test of 5 s bursts from 60 C, none reached the throttling point, and kernel time drifted by a median of 1.4% within a burst. The time limit never applies to vector add or layernorm, whose longest burst takes about 1.6 s.

In the layernorm run reported here, measurements started at 66 to 70 C and ended at 74 to 85 C. A second full run moved no value in its results table by more than 0.5%.

In the GEMM run, every measurement started at 60 C. The rounds ended at 66 to 85 C, and their medians were within 1% of each other. During tuning at 8,192, 12 of the 371 configurations ended at 86 C or above.

## Vector add results

Vector add is limited by memory bandwidth, so it checks the benchmark harness rather than the compilers. Throughput in GB/s, using the best configuration for each path and size:

| Elements | PyTorch | cuTile | Triton PTX, pointer | Triton PTX, descriptor | Triton Tile IR, pointer | Triton Tile IR, descriptor |
|---|---|---|---|---|---|---|
| 2^16 | 109.7 | 109.7 | 109.7 | 109.7 | 109.7 | 109.7 |
| 2^20 | 279.3 | 279.4 | 285.8 | 285.8 | 285.8 | 286.1 |
| 2^22 | 307.2 | 307.2 | 307.2 | 307.2 | 307.2 | 307.2 |
| 2^24 | 314.1 | 314.6 | 314.1 | 314.1 | 314.1 | 314.6 |
| 2^26 | 316.0 | 316.3 | 316.0 | 316.0 | 316.0 | 315.6 |

All paths reach about 316 GB/s at the largest size, roughly 94% of the card's ~336 GB/s specification. The spread at 2^20 is timer noise: those kernels take 43 to 45 us, and the timer moves in steps of about 1 us.

Pointer and descriptor loads match because memory bandwidth is the limit, so this says nothing yet about the tensor-of-pointer slowdown.

## Layernorm results

Layernorm normalizes each row of an fp16 matrix with 2^24 elements. Each row needs two reductions before it can be normalized, which gives the compilers something to differ on. Every path tries a whole-row kernel and a looped kernel, Triton tries pointer and descriptor loads for both, and tuning sweeps warps, occupancy and the latency hint where a path has them.

Effective throughput in GB/s, counting one read and one write per element, for the best variant of each path:

| Row length | PyTorch | Triton PTX | Triton Tile IR | cuTile |
|---|---|---|---|---|
| 1,024 | 286 | 312 | 311 | 308 |
| 2,048 | 223 | 312 | 308 | 312 |
| 4,096 | 211 | 311 | 311 | 311 |
| 8,192 | 205 | 309 | 309 | 309 |
| 16,384 | 204 | 304 | 305 | 303 |
| 32,768 | 202 | 298 | 298 | 293 |
| 65,536 | 205 | 162 | 206 | 178 |

- Up to 16,384 columns, the three compiled paths are within 2% of each other at each row length and close to the 316 GB/s of vector add, so memory bandwidth is the likely limit.
- At 32,768 columns, Triton Tile IR and PTX tie at 298 GB/s, and cuTile reaches 293.
- At 65,536 columns, Triton Tile IR matches PyTorch and runs 27% faster than PTX.

The result at 65,536 columns is measured but not yet explained.

### Why cuTile trailed at 32,768 columns

cuTile first reached 245 GB/s at 32,768 columns. Two details of how the whole-row kernel was written caused most of that gap, and fixing them raised it to 293:

- **The mask came after the load.** Computed below `ct.load`, the mask left each part of the row load in its own branch on sm_86, so a warp waited for one load before issuing the next. Moving that one independent line above the load removed the branches.
- **`w` and `b` had their own bounds.** cuTile passes each array's shape as a separate runtime value, so the compiler could not tell they were equal, and the last phase of the kernel kept about half as many loads in flight. Loading them with `ct.gather` under one shared mask fixed that.

Neither change does much alone. At the fixed settings of the investigation, the kernel went from 246 GB/s to 261 with the first change, 254 with the second, and 295 with both.

Why statement order changes the generated code is not known, because tileiras is closed. The branches appear on sm_80 to sm_89 and not on sm_90 or later, which were compiled but not run. [The investigation](investigations/cutile_32768/README.md) has the evidence, what was ruled out, and how this looks on tileiras 13.5.

### The latency hint

On Tile IR, Triton's `num_stages` becomes a `latency` hint on every load and store. It is the same hint that cuTile sets with `latency=`:

```text
load_view_tko weak %sview[...] optimization_hints = <sm_86 = {latency = 3}> : ...
```

Triton sends 3 by default. cuTile sends no hint unless asked, which is not always the same as 3: the two gave different SASS on looped kernels with 1,024-element tiles. On the PTX backend, `num_stages` left the SASS of all four layernorm kernels unchanged at 1,024 and 32,768 columns with 4 and 8 warps, so only the Tile IR paths sweep it.

The hint has a large effect, and no single value is best. Best GB/s at each setting during tuning:

| Kernel | Row length | No hint | 1 | 3 | 5 | 10 |
|---|---|---|---|---|---|---|
| Triton Tile IR, whole row, pointer | 32,768 | - | 257 | 272 | 299 | 224 |
| Triton Tile IR, whole row, descriptor | 65,536 | - | 172 | 136 | 112 | 193 |
| Triton Tile IR, looped, pointer | 16,384 | - | 166 | 268 | 191 | 198 |
| cuTile, whole row | 32,768 | 271 | 253 | 270 | 293 | 224 |
| cuTile, looped | 16,384 | 191 | 186 | 265 | 247 | 167 |

This changes the comparison on long rows. At 32,768 columns, the best Triton Tile IR result during tuning is 275 GB/s at the default of 3 and 299 with the hint swept, which is what lets it tie PTX. The best cuTile result there is 271 GB/s without a hint and 293 with one. Up to 16,384 columns, the hint moves each path's best result by less than 2%.

The hint also hid a kernel problem. At 16,384 columns, the cuTile whole-row kernel used to need one, reaching 251 GB/s without it and 303 with it. Since the fix described above, it reaches 303 to 305 at every setting.

## GEMM results

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

### Why cuBLAS is slowest at 1,024

The PyTorch profiler shows which kernel cuBLAS runs. At 1,024 it is `ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_32x3_nn`, which computes a 256×128 output tile per block. Its registers (218 per thread over 8 warps) and shared memory (72 KB) allow one block per SM at a time.

A 1,024×1,024 output has 32 such tiles, and the GPU has 30 SMs. So 30 blocks run, and then the last 2 run alone while 28 SMs sit idle. The kernel takes two rounds for 1.07 rounds of work, which uses about 53% of the GPU, and 14.7 TFLOPS is 54% of the 27.4 that cuBLAS reaches at 8,192. The tuned tile kernels use 128×32 tiles at this size, which gives 256 smaller blocks.

At 4,096, cuBLAS uses the same tile shape in 512 blocks, so the idle part of the last round costs at most about 5%. At 8,192 it switches to a 128×128 tile.

### cuTile's warp count on the largest tile

cuTile picks the warp count itself. On the largest tile, 128×256×64, it picked 4 warps. The fp32 accumulator alone then needs 256 registers per thread, more than the limit of 255, so the kernel spilled 1,112 bytes per thread to local memory and ran at 6.6 TFLOPS at 4,096. With `num_worker_warps=8` it used 222 registers, did not spill, and ran at 22.9 TFLOPS, 3.5 times faster.

The Triton paths use the tutorial's 8 warps on that tile, so cuTile gets 8 there as well. Every other tile keeps the compiler's choice, which was 4 warps on every tile inspected. The cuTile documentation describes `num_worker_warps` as a setting for warp-specialized kernels, but on sm_86 it set the block size directly. The tile never won for cuTile and stayed 9% behind its best at 8,192, so the change made the comparison fair without moving cuTile's results.

### Occupancy, pipeline depth and the latency hint

Occupancy 2 won for most Tile IR paths and sizes. Because 2 is the top of the sweep, occupancy 3 and 4 were tried at 4,096 on every tile within 4% of the best, and they never beat the best result at occupancy 2. To fit more blocks per SM, the compiler caps the registers per thread, at 168 for occupancy 3 and 128 for occupancy 4 with 128 threads, and it shrinks the shared-memory pipeline. Tiles whose accumulator no longer fits then spill. cuTile in TFLOPS, best over hints at 4,096:

| Tile | Occupancy 2 | Occupancy 3 | Occupancy 4 |
|---|---|---|---|
| 64×128×32 | 25.1 (160 regs, 48 KB) | 25.5 (146 regs, 28 KB) | 23.6 (128 regs, 20 KB, spills 40 B) |
| 128×128×32 | 25.8 (228 regs) | 20.2 (168 regs, spills 176 B) | 2.7 (128 regs, spills 504 B) |
| 64×256×32 | 26.7 (216 regs) | 17.8 (168 regs, spills 192 B) | 23.7 (one block per SM) |

Sometimes the compiler does not meet the request at all. 64×256×32 at occupancy 4 got one block per SM, and 128×256×64 builds the same at occupancy 1 to 4, also with one block per SM. Triton Tile IR behaves the same way, since both paths compile through tileiras.

On PTX, the number of pipeline stages is the matching trade-off, and 2 or 3 stages won. One stage, tried at 8,192 on the five closest tiles, was 16 to 29% slower: Triton then emits no asynchronous copies (`LDGSTS`), so every step along K waits for its loads. More stages cost shared memory. On 64×256×32, a fourth stage takes it from 40 to 60 KB, only one block fits per SM instead of two, and speed drops by about 10%.

The latency hint mattered less than for layernorm. On Triton Tile IR it moved each best result by at most 4%. On cuTile it mattered at 8,192, where the best result was 24.6 TFLOPS without a hint and 26.8 with a hint of 1.

### What the compilers produced

On the same tile, 64×256×32 at occupancy 2, cuTile and both Triton Tile IR kernels compile to nearly the same SASS at 8,192: 128 threads, 36 KB of shared memory, two blocks per SM, 128 `HMMA.16816.F32` tensor core instructions, 20 asynchronous copies and 32 `LDSM` matrix loads from shared memory. So Tile IR turns tensor-of-pointer loads into the same asynchronous copies as tile loads.

The one visible difference is the cache policy of those copies. The pointer kernel's copies skip L1 (`LDGSTS.E.BYPASS.128`), and those of the descriptor and cuTile kernels do not (`LDGSTS.E.128`). Triton PTX, inspected on 128×256×64, uses the bypassing form for both load styles.

Speed still differs on this tile: during tuning, cuTile reached 26.8 TFLOPS, Triton Tile IR 26.1 with pointer loads and 25.4 with descriptors. cuTile also uses the form that does not skip L1 and is the fastest of the three, so the cache policy alone does not explain the descriptor gap. On 128×256×64 the pattern repeats: the instruction mix is the same apart from the cache policy, and pointer loads are 6% faster. Finding the cause needs a closer comparison of the SASS, which is the next step of the tensor-of-pointer investigation.

One result is not yet explained. At 8,192, Triton PTX kernels with narrow tiles slowed down sharply, with both load styles and every stage count: 128×32×32 ran at 4.8 TFLOPS against 21.1 at 4,096, and 128×64×32 at about 10 against 25. The samples were steady and the clocks normal, so it is not a measurement problem, and Triton Tile IR slowed far less on the same tiles. These tiles never win at that size.

These results come from one laptop GPU and one compiler version. sm_86 has no TMA, so they do not predict how Tile IR behaves on Hopper or Blackwell.
