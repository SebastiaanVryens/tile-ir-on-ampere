from types import SimpleNamespace

import pytest

from tilebench.targets import get_target
from tilebench.worker import require_backend


def compiled_on(backend: str) -> SimpleNamespace:
    target = SimpleNamespace(backend=backend)
    return SimpleNamespace(metadata=SimpleNamespace(target=target))


def test_matching_backend_passes() -> None:
    require_backend(get_target("triton-tileir"), compiled_on("tileir"))


def test_silent_fallback_to_ptx_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="triton-tileir compiled on backend cuda"):
        require_backend(get_target("triton-tileir"), compiled_on("cuda"))


def test_non_triton_targets_have_nothing_to_prove() -> None:
    require_backend(get_target("cutile"), None)
