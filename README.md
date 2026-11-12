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
- [x] Tensor-of-pointer investigation

## Findings

- **GEMM: every path comes within 8% of cuBLAS from 2,048 up.** At 8,192, Triton PTX and Triton Tile IR with pointer loads reach 99% of cuBLAS, and cuTile reaches 97%. [GEMM results](docs/gemm.md)
- **cuBLAS is the slowest path at 1,024**, at 14.7 TFLOPS against 21.8 to 23.8 for the tile compilers. Its kernel makes 32 blocks of 256×128 for 30 SMs that fit one block each, so the last 2 blocks run alone and the GPU is about half used. [Why](docs/gemm.md#why-cublas-is-slowest-at-1024)
- **cuTile's own warp choice spilled on the largest GEMM tile.** With 4 warps on 128×256×64, the accumulator alone needs 256 registers per thread, over the limit of 255. With 8 warps it ran 3.5 times faster. [Details](docs/gemm.md#cutiles-warp-count-on-the-largest-tile)
- **Layernorm: Triton Tile IR is 27% faster than PTX at 65,536 columns**, and matches PyTorch. Up to 16,384 columns, the three compiled paths are within 2% of each other. The result at 65,536 is not yet explained. [Layernorm results](docs/layernorm.md)
- **Two details of how a kernel was written cost cuTile 16% at 32,768 columns:** the mask was computed after the load, and `w` and `b` had their own bounds. Fixing both took cuTile from 245 to 293 GB/s, next to Triton's 298. [Investigation](investigations/cutile_32768/README.md)
- **The latency hint is worth sweeping, and no single value is best.** On layernorm at 32,768 columns, it takes Triton Tile IR from 275 GB/s at the default to 299, which is what lets it tie PTX. [Details](docs/layernorm.md#the-latency-hint)
- **Tensor-of-pointer loads are not slow here, unless the compiler cannot tell that the sizes are multiples of 8.** At the benchmark sizes, pointer loads were as fast as descriptors or faster. At 4,088, a multiple of 8 but not of 16, every compiled path loses 34 to 84% of its throughput at 4,096, while cuBLAS loses 0.5%, because the compilers only record whether a size is divisible by 16. Telling them about the 8 brings each path back within 3 to 6%, in Triton and in cuTile. [Investigation](investigations/tensor_of_pointer/README.md)

## GEMM at a glance

Square fp16 matrices with fp32 accumulation, in TFLOPS, for the best configuration of each path and size:

| Size | PyTorch (cuBLAS) | cuTile | Triton PTX, pointer | Triton PTX, descriptor | Triton Tile IR, pointer | Triton Tile IR, descriptor |
|---|---|---|---|---|---|---|
| 1,024 | 14.7 | 22.5 | 23.8 | 23.8 | 22.6 | 21.8 |
| 2,048 | 26.9 | 26.2 | 27.1 | 26.5 | 26.1 | 25.5 |
| 4,096 | 27.2 | 26.6 | 27.1 | 26.2 | 26.4 | 25.7 |
| 8,192 | 27.4 | 26.7 | 27.2 | 26.3 | 27.0 | 25.2 |

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

## How it is measured

Every configuration is checked against PyTorch before it is timed. Tuning times each configuration briefly and keeps the fastest one for each variant and size, and three rounds then measure only those winners, so each reported value is the median of 300 launches that played no part in picking it. On a laptop GPU, clocks drop as soon as the load stops or the GPU passes 82 C, so kernels run in bursts under continuous load, starting from a cool GPU.

## More detail

- [Measurement and harness checks](docs/measurement.md): the timing method, the probe, and vector add
- [Layernorm results](docs/layernorm.md)
- [GEMM results](docs/gemm.md)
- [Investigation: why cuTile trailed at 32,768 columns](investigations/cutile_32768/README.md)
- [Investigation: tensor-of-pointer loads](investigations/tensor_of_pointer/README.md)

These results come from one laptop GPU and one compiler version. sm_86 has no TMA, so they do not predict how Tile IR behaves on Hopper or Blackwell.
