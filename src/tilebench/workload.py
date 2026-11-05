"""What every benchmarked kernel provides."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

import torch

Size = tuple[int, ...]
Config = dict[str, Any]
Inputs = tuple[torch.Tensor, ...]


class Prepared(NamedTuple):
    """A launch with its output already allocated, so timing excludes allocation.

    Triton launches must return the compiled kernel, which proves the backend used.
    cuTile launches return nothing, so tile_kernel holds the kernel and its arguments.
    """

    launch: Callable[[], Any]
    output: torch.Tensor
    tile_kernel: tuple[Any, tuple[Any, ...]] | None = None


class Impl(Protocol):
    """One implementation module, such as kernels/vector_add/triton_impl.py."""

    def configs(self, backend: str, size: Size) -> tuple[Config, ...]: ...

    def prepare(self, inputs: Inputs, config: Config) -> Prepared: ...


@dataclass(frozen=True)
class Workload:
    name: str
    sizes: tuple[Size, ...]
    check_sizes: tuple[Size, ...]
    make_inputs: Callable[[Size], Inputs]
    reference: Callable[..., torch.Tensor]
    work: Callable[[Size], float]
    rate_unit: str
    rate_scale: float
    rtol: float
    atol: float
