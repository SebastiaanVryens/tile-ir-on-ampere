"""cuTile probe cases, one per process, launched by probes/run.py."""

import json
import sys
from collections.abc import Callable
from typing import Any

import cuda.tile as ct
import torch

SIZE = 512
TILE_M, TILE_N, TILE_K = 64, 64, 32
ADD_TILE = 1024


@ct.kernel
def add_kernel(a, b, c, tile: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile,))
    b_tile = ct.load(b, index=(pid,), shape=(tile,))
    ct.store(c, index=(pid,), tile=a_tile + b_tile)


@ct.kernel
def matmul_kernel(a, b, c, tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]):
    bid_m, bid_n = ct.bid(0), ct.bid(1)
    num_k = ct.num_tiles(a, axis=1, shape=(tm, tk))
    acc = ct.full((tm, tn), 0, dtype=ct.float32)
    for k in range(num_k):
        a_tile = ct.load(a, index=(bid_m, k), shape=(tm, tk))
        b_tile = ct.load(b, index=(k, bid_n), shape=(tk, tn))
        acc = ct.mma(a_tile, b_tile, acc)
    ct.store(c, index=(bid_m, bid_n), tile=ct.astype(acc, c.dtype))


def summarize() -> dict[str, Any]:
    major, minor = torch.cuda.get_device_capability()
    return {"backend": "cutile", "arch": major * 10 + minor}


def run_add() -> dict[str, Any]:
    n = 1 << 20
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    c = torch.empty_like(a)
    grid = (ct.cdiv(n, ADD_TILE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, add_kernel, (a, b, c, ADD_TILE))
    torch.testing.assert_close(c, a + b)
    return summarize()


def run_matmul() -> dict[str, Any]:
    torch.manual_seed(0)
    a = torch.randn(SIZE, SIZE, device="cuda", dtype=torch.float16)
    b = torch.randn(SIZE, SIZE, device="cuda", dtype=torch.float16)
    c = torch.empty_like(a)
    grid = (SIZE // TILE_M, SIZE // TILE_N, 1)
    args = (a, b, c, TILE_M, TILE_N, TILE_K)
    ct.launch(torch.cuda.current_stream(), grid, matmul_kernel, args)
    torch.testing.assert_close(c.float(), a.float() @ b.float(), rtol=2e-2, atol=1e-1)
    return summarize()


CASES: dict[str, Callable[[], dict[str, Any]]] = {"add": run_add, "matmul": run_matmul}


if __name__ == "__main__":
    print(json.dumps(CASES[sys.argv[1]]()))
