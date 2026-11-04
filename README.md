# Tile IR on Ampere: cuTile and Triton compared

This project writes the same GPU kernels three ways: cuTile Python, Triton on the PTX backend, and Triton on the CUDA Tile IR backend. PyTorch is the reference for correctness and performance.
The goal is to understand how each path compiles, not only which one is faster.

## Status

- [x] Pinned environment
- [x] Probe: every path compiles and runs correctly on sm_86
- [ ] Benchmark harness
- [ ] Kernels: vector add, fused softmax or layernorm, tiled GEMM
- [ ] Tensor-of-pointer investigation

There are no performance results yet.

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
```

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
