"""Every target must match PyTorch on every workload, including awkward sizes."""

import shutil
import subprocess
import sys

import pytest

from tilebench.kernels import WORKLOAD_NAMES
from tilebench.targets import TARGETS, Target, worker_env

TIMEOUT_S = 600

pytestmark = pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="needs a GPU")


@pytest.mark.parametrize("workload", WORKLOAD_NAMES)
@pytest.mark.parametrize("target", TARGETS, ids=lambda target: target.name)
def test_matches_reference(target: Target, workload: str) -> None:
    command = [
        sys.executable,
        "-m",
        "tilebench.worker",
        "--target",
        target.name,
        "--workload",
        workload,
        "--check",
    ]
    proc = subprocess.run(
        command, env=worker_env(target), capture_output=True, text=True, timeout=TIMEOUT_S
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
