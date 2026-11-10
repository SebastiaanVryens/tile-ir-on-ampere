# Layernorm results

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

- Up to 16,384 columns, the three compiled paths are within 2% of each other at each row length and close to the 316 GB/s of [vector add](measurement.md#vector-add-results), so memory bandwidth is the likely limit.
- At 32,768 columns, Triton Tile IR and PTX tie at 298 GB/s, and cuTile reaches 293.
- At 65,536 columns, Triton Tile IR matches PyTorch and runs 27% faster than PTX.

The result at 65,536 columns is measured but not yet explained.

## Why cuTile trailed at 32,768 columns

cuTile first reached 245 GB/s at 32,768 columns. Two details of how the whole-row kernel was written caused most of that gap, and fixing them raised it to 293:

- **The mask came after the load.** Computed below `ct.load`, the mask left each part of the row load in its own branch on sm_86, so a warp waited for one load before issuing the next. Moving that one independent line above the load removed the branches.
- **`w` and `b` had their own bounds.** cuTile passes each array's shape as a separate runtime value, so the compiler could not tell they were equal, and the last phase of the kernel kept about half as many loads in flight. Loading them with `ct.gather` under one shared mask fixed that.

Neither change does much alone. At the fixed settings of the investigation, the kernel went from 246 GB/s to 261 with the first change, 254 with the second, and 295 with both.

Why statement order changes the generated code is not known, because tileiras is closed. The branches appear on sm_80 to sm_89 and not on sm_90 or later, which were compiled but not run. [The investigation](../investigations/cutile_32768/README.md) has the evidence, what was ruled out, and how this looks on tileiras 13.5.

## The latency hint

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
