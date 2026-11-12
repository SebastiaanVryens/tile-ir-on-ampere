# Tensor-of-pointer loads on sm_86

The CUDA Tile IR README warns that kernels using the "legacy tensor-of-pointer load/store APIs exhibit poor performance". In Triton, a tensor-of-pointer load reads from a tensor of addresses, such as `a_ptr + offsets`. A tensor descriptor describes the whole array instead, and the kernel loads tiles from it by index. This investigation checks the warning on an RTX 3060 Laptop GPU (sm_86), which has no TMA.

At the benchmark sizes, the warning does not hold: pointer loads were as fast as descriptors or faster. It holds when the compiler cannot tell that the sizes are multiples of 8. At 4,088, every compiled path falls back to 2-byte loads, cuTile included, and Triton's pointer kernel on Tile IR loses the most. Telling the compilers that the sizes are multiples of 8 brings the fast code back, and there is a way to do so in both Triton and cuTile.

Unless noted, figures are for an fp16 GEMM with fp32 accumulation, on Triton 3.7.1 from the Tile IR wheel (tileiras 13.4.59) for both Triton backends, and cuda-tile 1.7.0 (tileiras 13.4.92). Every kernel is checked against the reference before it is timed, and every burst starts at 60 C, as in the benchmark.

## Result

TFLOPS, best over every setting of the tiles 128×256×64 and 128×128×32, with 20 launches per setting as in the benchmark's tuning:

| Path | 4,096 | 4,088 | Change | 4,088, told the sizes are multiples of 8 |
|---|---|---|---|---|
| PyTorch (cuBLAS) | 26.9 | 26.7 | -0.5% | - |
| cuTile | 25.4 | 7.3 | -71% | 24.8 |
| Triton PTX, pointer | 27.1 | 14.2 | -48% | 25.9 |
| Triton PTX, descriptor | 26.4 | 17.5 | -34% | 25.3 |
| Triton Tile IR, pointer | 26.6 | 4.2 | -84% | 25.0 |
| Triton Tile IR, descriptor | 25.4 | 9.6 | -62% | 24.3 |

- At 4,088, the best build of every compiled path loads and stores 2 bytes at a time (`LDG.E.U16`, `STG.E.U16`) instead of with 16-byte asynchronous copies, and almost every build spills.
- The same pointer code loses 84% on Tile IR and 48% on PTX. On Tile IR, the descriptor kernel keeps more than twice the pointer kernel's speed. This is the form in which the warning holds on this GPU.
- The last column passes `K // 8 * 8` in place of `K` in Triton, and loads from sliced views in cuTile, as described under Workarounds. Every path then comes within 3 to 6% of its speed at 4,096.

## At the benchmark sizes

Best pointer kernel against best descriptor kernel, from the benchmark's rounds, on each Triton backend:

- **Vector add:** within 0.2%.
- **Layernorm:** descriptors were never more than 1.4% faster. Pointer loads were faster by 7 to 12% in two spots on Tile IR, and by 19% at 65,536 columns on PTX.
- **GEMM:** pointer loads were faster on both backends from 2,048 up, by 2.5 to 7.3% on Tile IR and about 3% on PTX.

Pointer code was more sensitive to its settings, though. Comparing the same settings during tuning, descriptors were faster in most pairs of Tile IR's looped layernorm up to 8,192 columns, by a median of 5 to 10%. Tuning found settings where the pointer kernel caught up.

## Why pointer loads win the GEMM

The repository's pointer kernel follows Triton's tutorial, which is not naive: it wraps the row and column indices with `% M` and `% N`, so only the K dimension needs a mask. Without TMA, the bounds checks of a descriptor load become ordinary instructions in the loop, on every dimension. The main loop at 4,096 on 128×256×64, with 8 warps and 3 stages, per step along K:

| Kernel | Instructions | Comparisons (`ISETP`) | Selects (`SEL`) | Address math (`LEA`) |
|---|---|---|---|---|
| Triton PTX, pointer | 241 | 13 | 11 | 1 |
| Triton PTX, descriptor | 330 | 18 | 26 | 16 |
| Triton Tile IR, pointer | 270 | 1 | 0 | 1 |
| Triton Tile IR, descriptor | 315 | 34 | 20 | 5 |

All four issue the same 128 tensor core instructions and 12 asynchronous copies per step. Triton on PTX rewrites descriptor loads into masked asynchronous copies like the pointer kernel's, with extra address arithmetic. tileiras keeps them as tile loads, and reduces the pointer kernel's mask along K to a single comparison.

The extra work fits the pointer kernels' lead, but does not prove it, and one tile shows its limits. On 64×256×32 at occupancy 2, cuTile's loop is longer than Triton's pointer loop, at 199 instructions per step with 29 comparisons against 187 with 1, yet cuTile was the fastest of the three on that tile during the benchmark's tuning. Instruction count alone does not decide the speed.

The GEMM results noted one more difference: the pointer kernel's copies on Tile IR skip L1 (`LDGSTS.E.BYPASS.128`), and the descriptor kernel's do not. This could not be tested from Triton. Asking for L1 caching (`cache_modifier=".ca"`) gives the same SASS on both backends.

## Pointer code without the tutorial's trick

Written the plain way, with masks on both dimensions and no wrapping, the pointer kernel was 53% slower on Tile IR at the setting above with occupancy 1, at 17.2 against 26.3 TFLOPS in rounds. It ran out of registers: its main loop has 451 instructions per step, among them 49 loads and 34 stores to local memory.

That looked like the warning, but the sweep over every setting showed one setting falling over. On Tile IR, 7 of the 88 builds of each kernel spill, always at the limit of 255 registers, and the second mask only changes which ones. Measured in rounds, the best plain kernel ran within 1.3% of the best tutorial-style kernel, at 26.0 against 26.3 TFLOPS. On PTX, the plain kernel was 0 to 5% slower on most tiles, and 1.9% slower at its best setting in rounds.

## Why 4,088 is slow

A 16-byte asynchronous copy moves 8 fp16 values, and its address must be a multiple of 16 bytes. 4,088 is 8 × 511, so every row of the matrices starts 16-byte aligned. But when Triton and cuTile compile a kernel, they record whether each size is divisible by 16, and nothing finer.

**Triton** marks integer arguments divisible by 16. At 4,088 it marks nothing, so the compiler cannot prove that rows start aligned. Turning off that specialization (`do_not_specialize`) at 4,096 gives the same fallback, at 4.1 TFLOPS for the pointer kernel on Tile IR and 14.4 on PTX, so it is the recorded divisibility that matters, not the size.

**cuTile** derives a signature from the arguments, with what each array's shape, strides and base address are divisible by:

| Size | Shape | Strides | Base address |
|---|---|---|---|
| 4,096 | 16, 16 | 8, 1 | 16 bytes |
| 4,088 | 1, 1 | 8, 1 | 16 bytes |

So cuTile knows at 4,088 that every row starts 16-byte aligned. Only the divisibility of the shapes is lost, and that alone produces the fallback: compiled ahead of time with the shapes declared divisible by 8, the 4,088 kernel has the same SASS as the 4,096 one, line for line, on both tiles. The rule for PyTorch tensors is in compiled code, but the JAX path spells it out in Python: `_DIVISOR_16 if d % _DIVISOR_16 == 0 else 1`.

**Descriptors** do not avoid the fallback, although Triton documents that their strides must be multiples of 16 bytes. Aligned strides make every row start aligned, but say nothing about where the array ends. The cuTile result suggests that the edge is what the compilers cannot handle with wide copies, but this was not verified.

## Workarounds

Each of these only tells the compiler something, and is only correct if the caller has checked that the sizes really are multiples of 8. Otherwise the kernel computes on smaller sizes or reads out of bounds.

**Triton.** Three ways to say that `K` is a multiple of 8, each applied to `M`, `N` and `K`. TFLOPS at 4,088, at the best setting:

| Way | PTX, pointer | Tile IR, pointer |
|---|---|---|
| `tl.multiple_of(K, 8)` | no effect | no effect |
| `tl.assume(K % 8 == 0)` | no effect, 14.1 | 24.8 |
| `K // 8 * 8` in place of `K` | 25.9 | 25.0 |

`tl.multiple_of` is the language's divisibility hint, but on a kernel argument it is dropped without an error: the Triton IR has no trace of it, and the SASS is the same as without it, so it was not timed. `tl.assume` reaches tileiras but not Triton's own PTX path. `K // 8 * 8` works on both, because the compiler can derive the divisibility from the arithmetic. The descriptor kernels respond the same way.

**cuTile.** `ct.assume_divisible_by` takes integer scalars, and assuming it on `a.shape[0]` alone changes nothing, because `ct.load` checks the array's own shape. What works is to slice each array to its own length, with that length assumed divisible by 8, and load from the slices:

```python
m = ct.assume_divisible_by(a.shape[0], 8)
k = ct.assume_divisible_by(a.shape[1], 8)
a_view = a.slice(0, 0, m).slice(1, 0, k)
```

The slices cover the whole array, so only what the compiler knows changes. At 4,088 the kernel then builds the same copies and register count as at 4,096, and runs at 24.8 TFLOPS against 25.4.

## Not known

- Why Triton's pointer kernel falls furthest on Tile IR. At 4,088, its best build spills about as much as the PTX one, 368 bytes against 352, yet runs 3.4 times slower.
- Whether recording only divisibility by 16 is deliberate. A finer rule would compile more variants of the same kernel, which may be the reason.
- The 3 to 6% that remain with the workarounds, where cuBLAS loses 0.5% at 4,088.
- How any of this looks with TMA, on Hopper or Blackwell.

## Running it

From the activated venv. The scripts that build kernels start themselves once per target, in that target's environment, and every script prints its tables:

```bash
python investigations/tensor_of_pointer/sizes.py
```

| Script | What it shows | Time |
|---|---|---|
| `sizes.py` | The result table, with Triton's specialization off and each workaround | About 11 minutes |
| `signature.py` | cuTile's signatures and the ahead-of-time builds | Under a minute |
| `hints.py` | The three ways to tell Triton, with their Triton IR and SASS | Under a minute |
| `main_loop.py` | The main loops per step along K on both tiles named above, and the L1 request | Under a minute |
| `masked.py` | The plain pointer kernel at every setting, then in rounds | About 9 minutes |
| `styles.py` | Pointer against descriptor in saved benchmark runs; no GPU needed | Seconds |

The times are for a second run. The first one takes longer while every kernel compiles.

The two timed scripts print the SM clock and power that the harness read after each burst. In the runs reported here, their medians were 1,760 to 1,905 MHz and, apart from PyTorch's two short bursts, 127 to 130 W. A run that shows much lower values was held back, for example by another program using the GPU or by a laptop power mode, and should be repeated. `--keep DIR` also saves the raw records of each target.

`styles.py` reads the files that `python -m tilebench` writes to `results/`. The figures above come from one run of each kernel:

```bash
python investigations/tensor_of_pointer/styles.py results/VECTOR_ADD_RUN.json --workloads vector_add
python investigations/tensor_of_pointer/styles.py results/LAYERNORM_RUN.json --workloads layernorm
python investigations/tensor_of_pointer/styles.py results/GEMM_RUN.json --workloads gemm
```

`kernels.py` holds the Triton variants, `cutile_kernels.py` the cuTile one, and `common.py` runs the targets and reads the SASS.
