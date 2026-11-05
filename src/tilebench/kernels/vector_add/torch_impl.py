"""PyTorch vector add, the performance reference."""

import torch

from tilebench.workload import Config, Inputs, Prepared, Size


def configs(backend: str, size: Size) -> tuple[Config, ...]:
    return ({},)


def prepare(inputs: Inputs, config: Config) -> Prepared:
    a, b = inputs
    out = torch.empty_like(a)
    return Prepared(lambda: torch.add(a, b, out=out), out)
