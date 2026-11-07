"""PyTorch matrix multiply through cuBLAS, the performance reference."""

import torch

from tilebench.workload import Config, Inputs, Prepared, Size

# By default PyTorch lets cuBLAS reduce partial sums in fp16, which no kernel here does. That
# put 6% of the results at 256x256x8192 outside the tolerance. Turning it off keeps every
# path on the same math, and moved speed at the benchmark sizes by at most 0.5%.
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def configs(backend: str, size: Size) -> tuple[Config, ...]:
    return ({},)


def prepare(inputs: Inputs, config: Config) -> Prepared:
    a, b = inputs
    c = torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)
    return Prepared(lambda: torch.mm(a, b, out=c), c)
