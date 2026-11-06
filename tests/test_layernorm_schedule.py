"""The cuTile whole-row kernel must keep its load schedule, which is worth 19% at 32768 columns.

Both properties depend on how the kernel is written rather than on what it computes, so a
harmless looking edit can lose them. cutile_impl.py explains the two changes they guard.
"""

import shutil

import pytest
import torch

from tilebench.artifacts import from_cutile
from tilebench.cubin import read_kernel
from tilebench.kernels.layernorm import cutile_impl, make_inputs
from tilebench.schedule import LoadSchedule, read_schedule

COLS = 32768

pytestmark = pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="needs a GPU")


def schedule_of(latency: int) -> LoadSchedule:
    # Sixteen rows, because the row count's divisibility is part of what cuTile specializes on.
    inputs = make_inputs((16, COLS))
    config = {
        "style": "whole",
        "tile": COLS,
        "occupancy": 1,
        "num_worker_warps": 8,
        "latency": latency,
    }
    prepared = cutile_impl.prepare(inputs, config)
    assert prepared.tile_kernel is not None
    return read_schedule(read_kernel(from_cutile(*prepared.tile_kernel).cubin).sass)


@pytest.mark.parametrize(("latency", "tail_loads"), ((3, 8), (5, 12)))
def test_whole_row_loads_overlap(latency: int, tail_loads: int) -> None:
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("the branches only appear on sm_80 to sm_89")
    schedule = schedule_of(latency)
    assert schedule.branch_regions == 0
    assert schedule.first_peak >= 7
    assert schedule.last_peak >= tail_loads
