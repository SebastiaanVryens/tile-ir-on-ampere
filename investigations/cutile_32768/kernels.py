"""The cuTile whole-row kernel with each change on a switch.

With both switches on this is the repository kernel. With both off it is the kernel as it
was before this investigation, which is the only place that version still exists.
"""

from collections.abc import Callable

import cuda.tile as ct

ZERO = ct.PaddingMode.ZERO


def layernorm(
    latency: int,
    mask_first: bool,
    shared_mask: bool,
    rounding: ct.RoundingMode | None = None,
) -> Callable[..., None]:
    def whole(x, w, b, y, eps, tile: ct.Constant[int]):
        bid = ct.bid(0)
        n = x.shape[1]
        if mask_first:
            cols = ct.arange(tile, dtype=ct.int32)
            mask = cols < n
        row = ct.load(x, index=(bid, 0), shape=(1, tile), padding_mode=ZERO, latency=latency)
        row = ct.astype(row, ct.float32)
        mean = ct.truediv(ct.sum(row, axis=1), n, rounding_mode=rounding)
        if not mask_first:
            cols = ct.arange(tile, dtype=ct.int32)
            mask = cols < n
        centered = ct.where(mask, row - mean, 0)
        variance = ct.truediv(ct.sum(centered * centered, axis=1), n, rounding_mode=rounding)
        rstd = ct.truediv(1, ct.sqrt(variance + eps), rounding_mode=rounding)
        if shared_mask:
            w_tile = ct.gather(w, cols, mask=mask, check_bounds=False, latency=latency)
            b_tile = ct.gather(b, cols, mask=mask, check_bounds=False, latency=latency)
        else:
            w_tile = ct.load(w, index=(0,), shape=(tile,), padding_mode=ZERO, latency=latency)
            b_tile = ct.load(b, index=(0,), shape=(tile,), padding_mode=ZERO, latency=latency)
        result = centered * rstd * ct.astype(w_tile, ct.float32) + ct.astype(b_tile, ct.float32)
        ct.store(y, index=(bid, 0), tile=ct.astype(result, y.dtype), latency=latency)

    return whole
