# Measurement and harness checks

## How it is measured

`python -m tilebench` has two stages. Tuning times every configuration briefly and keeps the fastest one for each variant and size. Three rounds then measure only those winners, so the reported numbers do not come from the timings that picked them. Each reported value is the median of 300 launches.

Every configuration is checked against PyTorch before it is timed, and the L2 cache is cleared before each launch. A laptop GPU needs two more precautions:

- **Continuous load.** Clocks only stay up under steady load. Timing one kernel at a time with about 100 ms gaps ran near 580 MHz and read up to 2.5x too slow. So kernels are compiled and checked first, then timed in bursts: one 0.75 s warmup, then launches back to back.
- **A cool start.** The median SM clock holds at 1950 MHz up to 82 C, then falls to 1575 MHz at 86 C. Each burst waits until the GPU is at 70 C or below.
- **Shorter, cooler bursts for GEMM.** A burst is normally 640 launches, which at 8,192 would be 26 s of GEMM. Under cuBLAS at that size, the GPU reaches its 130 W power limit within a second, warms by about 1 C per second, and slows for heat at 86 to 87 C after about 30 s. So a burst also ends after 5 s, and GEMM bursts wait for 60 C. In a test of 5 s bursts from 60 C, none reached the throttling point, and kernel time drifted by a median of 1.4% within a burst. The time limit never applies to vector add or layernorm, whose longest burst takes about 1.6 s.

In the layernorm run reported in [the layernorm results](layernorm.md), measurements started at 66 to 70 C and ended at 74 to 85 C. A second full run moved no value in its results table by more than 0.5%.

In the GEMM run, every measurement started at 60 C. The rounds ended at 66 to 85 C, and their medians were within 1% of each other. During tuning at 8,192, 12 of the 371 configurations ended at 86 C or above.

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

Pointer and descriptor loads match because memory bandwidth is the limit, so vector add says nothing about the tensor-of-pointer question. [That investigation](../investigations/tensor_of_pointer/README.md) takes it up.
