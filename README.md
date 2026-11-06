# Tile IR on Ampere: cuTile and Triton compared

This project writes the same GPU kernels three ways: cuTile Python, Triton on the PTX backend, and Triton on the CUDA Tile IR backend. PyTorch is the reference for correctness and performance.
The goal is to understand how each path compiles, not only which one is faster.

## Status

- [x] Pinned environment
- [x] Probe: every path compiles and runs correctly on sm_86
- [x] Benchmark harness
- [x] Kernel: vector add
- [x] Kernel: layernorm
- [ ] Kernel: tiled GEMM
- [ ] Tensor-of-pointer investigation

Vector add validates the harness. Layernorm is the first kernel where the paths differ, on rows of 32,768 columns and longer.

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

`python -m tilebench` prints a comparison table and saves the raw timings in `results/`. A full run takes about 20 minutes, and longer the first time while every kernel compiles.

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

In the run reported here, measurements started at 67 to 70 C and ended at 72 to 84 C. A second full run of the layernorm benchmark moved no value in its results table by more than 0.5%.

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
| 1,024 | 285 | 312 | 311 | 308 |
| 2,048 | 223 | 312 | 308 | 312 |
| 4,096 | 211 | 311 | 311 | 309 |
| 8,192 | 205 | 311 | 309 | 306 |
| 16,384 | 204 | 303 | 305 | 303 |
| 32,768 | 202 | 298 | 298 | 245 |
| 65,536 | 205 | 162 | 206 | 174 |

- Up to 16,384 columns, the three compiled paths are within 2% of each other at each row length and close to the 316 GB/s of vector add, so memory bandwidth is the likely limit.
- At 32,768 columns, Triton Tile IR and PTX tie at 298 GB/s, while cuTile reaches 245.
- At 65,536 columns, Triton Tile IR matches PyTorch and runs 27% faster than PTX.

The last two results are measured but not yet explained.

### The latency hint

On Tile IR, Triton's `num_stages` becomes a `latency` hint on every load and store. It is the same hint that cuTile sets with `latency=`:

```text
load_view_tko weak %sview[...] optimization_hints = <sm_86 = {latency = 3}> : ...
```

Triton sends 3 by default. cuTile sends no hint unless asked, which is not always the same as 3: the two gave different SASS on looped kernels with 1,024-element tiles. On the PTX backend, `num_stages` left the SASS of all four layernorm kernels unchanged at 1,024 and 32,768 columns with 4 and 8 warps, so only the Tile IR paths sweep it.

The hint has a large effect, and no single value is best. Best GB/s at each setting during tuning:

| Kernel | Row length | No hint | 1 | 3 | 5 | 10 |
|---|---|---|---|---|---|---|
| Triton Tile IR, whole row, pointer | 32,768 | - | 256 | 270 | 298 | 224 |
| Triton Tile IR, whole row, descriptor | 65,536 | - | 171 | 135 | 112 | 193 |
| Triton Tile IR, looped, pointer | 16,384 | - | 166 | 267 | 190 | 200 |
| cuTile, whole row | 16,384 | 251 | 303 | 252 | 302 | 302 |
| cuTile, looped | 16,384 | 192 | 188 | 265 | 246 | 167 |

This changes the comparison on long rows. At 32,768 columns, the best Triton Tile IR result is 274 GB/s at the default of 3 and 298 with the hint swept, which is what lets it tie PTX. At 16,384 columns, the best cuTile result is 251 GB/s without a hint and 303 with one. Up to 8,192 columns, the hint moves each path's best result by less than 2%.

These results come from one laptop GPU and one compiler version. sm_86 has no TMA, so they do not predict how Tile IR behaves on Hopper or Blackwell.
