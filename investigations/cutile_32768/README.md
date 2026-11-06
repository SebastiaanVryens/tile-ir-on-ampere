# Why cuTile trailed at 32,768 columns

The benchmark put cuTile's whole-row layer norm at 245 GB/s, where Triton Tile IR reached 298, on an RTX 3060 Laptop GPU (sm_86). Two details of how the kernel was written caused most of that gap. Both are now fixed in `cutile_impl.py`, and cuTile reaches 293 in the benchmark.

Unless noted, figures are for 512 x 32,768 fp16 with one row per block, 8 warps and occupancy 1, on cuda-tile 1.7.0 and tileiras 13.4.92. These fixed settings give 295, where the benchmark's tuning settles on a configuration that gives 293.

## Result

Best GB/s across latency hints 1 to 10:

| Kernel | IEEE division | Full division |
|---|---|---|
| Original | 246 | 251 |
| Mask first | 261 | 266 |
| Shared `w`/`b` mask | 254 | 255 |
| Both | **295** | **295** |

Neither change does much alone, because each one removes a different limit. The schedule at hint 5 shows why:

| Kernel | Branch regions | Row loads in flight | Tail loads in flight | GB/s |
|---|---|---|---|---|
| Original | 16 | 1 | 7 | 237 |
| Mask first | 0 | 7 | 7 | 256 |
| Shared `w`/`b` mask | 16 | 1 | 12 | 240 |
| Both | 0 | 7 | 12 | 295 |
| Triton descriptor | 0 | 7 | 12 | 298 |
| Triton pointer | 0 | 7 | 14 | 299 |

Loads in flight are per warp, counted from the SASS by `tilebench.schedule`. They are upper bounds, not profiler measurements. The GB/s figures are measured.

## The two causes

**The mask was computed after the row load.** With `mask = arange < n` below `ct.load`, tileiras puts each part of the row load in its own branch region, with the first use of the data inside it. A warp then waits for one load before it issues the next. Moving that one independent line above the load gives plain predicated loads.

**`w` and `b` had their own bounds.** cuTile passes each array's shape as a separate runtime value, so the compiler cannot tell that the shapes of `x`, `w` and `b` are equal, and the tail stays near 7 loads in flight. Loading `w` and `b` with `ct.gather` under one shared mask raises that to 12. This has the same structure as Triton's pointer kernel, and it makes the same assumption: `w` and `b` must have `n` elements.

The second cause was tested in both directions. Sharing the bounds in cuTile's IR raises the tail from 7 to 13, and giving Triton's descriptor kernel separate bounds lowers it from 12 to 7.

## Ruled out

- **The compiler build.** cuTile and Triton use different tileiras builds, 13.4.92 and 13.4.59. Both produce the same instructions from the same bytecode.
- **IR features.** Tokens, partition against strided views, index alignment, per-target hints and a padded store each gave identical SASS. i64 shapes changed the code but not the branches or the tail.
- **1D against 2D compute.** Reshaping the row to 1D gave the same schedule counts.
- **Division, on 13.4.** This was a wrong first theory. Triton with IEEE division is just as fast, at 299 and 298, and full division adds nothing once both fixes are in.

## Not known

- Why the position of an independent statement changes the generated code. tileiras is closed and has no pass dump, so this is observed from the outside.
- Why separate bounds limit the tail. The effect is confirmed in both directions, the mechanism is not.
- The last 4 GB/s to Triton's pointer kernel at hint 5.

## Other compilers and GPUs

**tileiras 13.5.37.** The two causes remain: the original kernel still has its 13 to 16 branch regions, and the fix is still worth 16%. Division now matters indirectly, through register pressure:

| Kernel on 13.5.37 | Best GB/s |
|---|---|
| cuTile original | 249 |
| cuTile with both fixes, IEEE division | 290 |
| cuTile with both fixes, full division | 298 |
| Triton descriptor, approximate division | 298 |
| Triton descriptor, IEEE division | 275 |
| Triton pointer, either division | 299 |

At hint 5, IEEE division adds 6 spills to the cuTile kernel and 18 to Triton's descriptor kernel, and none to Triton's pointer kernel. cuda-tile 1.8.0 gave the same schedules and speeds as 1.7.0 on both compilers.

**Other GPUs.** In a cut-down kernel where only the mask position differs, the branch regions appear on sm_80, sm_86 and sm_89, and disappear when the mask moves first. On sm_90, sm_100 and sm_120 the count is 2 either way. Those targets were compiled but not run, as this machine only has sm_86.

## A compile failure found on the way

`ct.load` with `check_bounds=False` and a padding mode fails on tileiras 13.4 with a bare `failed to compile Tile IR program`, although the docs say the padding mode is ignored. The frontend keeps the padding on the view, and tileiras 13.4 refuses a padded view with any `inbounds` promise, on every target tried. Leaving the padding mode out avoids it.

tileiras 13.5.37 accepts the same bytecode, and the result matches the unpadded SASS, so this is fixed upstream. The same refusal was reported independently in [dawn-lang #686](https://github.com/dawnop/dawn-lang/pull/686).

## Running it

From the activated venv, each script prints its own tables and writes nothing:

```bash
bash investigations/cutile_32768/run.sh factorial.py
```

| Script | What it shows | Needs a GPU to run kernels |
|---|---|---|
| `factorial.py` | Both fixes and division, in every combination, at every hint | Yes |
| `reverse_division.py` | Triton with and without IEEE division | Yes |
| `ir_edits.py` | Tile IR round trip, the compiler builds, and every one-feature IR edit | No, but one is needed to build the kernels |
| `gpu_generations.py` | Whether the mask position matters on other GPUs | No, same as above |
| `padding.py` | The `check_bounds` failure, in every padding mode and API form | Yes |

`tile_ir.py` turns Tile IR text back into bytecode, using the assembler in Triton's Tile IR build. The round trip reproduces each frontend's SASS exactly, which is what makes single-line IR edits possible. The edits use the value names cuda-tile 1.7.0 prints, and each one asserts that it changed the text.

To repeat a script on tileiras 13.5, install a complete toolchain beside the venv. A 13.5 tileiras running against 13.4's `nvvm` fails with the same bare error once `--lineinfo` is on.

```bash
export TREE=$HOME/cu135
pip install --no-deps --target "$TREE" nvidia-cuda-tileiras==13.5.37 nvidia-cuda-nvcc==13.5.37 \
    nvidia-nvvm==13.5.37 nvidia-cuda-cuobjdump==13.5.37 nvidia-cuda-nvdisasm==13.5.37
export TOOL_BIN=$TREE/nvidia/cu13/bin
EXTRA_PYTHONPATH=$TREE PATH=$TOOL_BIN:$PATH TRITON_TILEIRAS_PATH=$TOOL_BIN \
    bash investigations/cutile_32768/run.sh factorial.py
```
