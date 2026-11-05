"""PyTorch layer norm, the performance reference."""

from typing import Any

from tilebench.kernels.layernorm import reference
from tilebench.workload import Config, Inputs, Prepared, Size


def configs(backend: str, size: Size) -> tuple[Config, ...]:
    return ({},)


def prepare(inputs: Inputs, config: Config) -> Prepared:
    # F.layer_norm allocates its result, but that only costs host time. The out variant
    # of native_layer_norm avoids the allocation and runs about 2x slower on the GPU.
    def launch() -> Any:
        return reference(*inputs)

    return Prepared(launch, launch())
