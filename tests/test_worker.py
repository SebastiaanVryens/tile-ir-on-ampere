from types import SimpleNamespace

import pytest

from tilebench.targets import get_target
from tilebench.worker import bursts, by_size, require_backend, selected


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


def test_plans_group_by_size_in_order() -> None:
    plan = [([2, 1024], {"block": 1}), ([4, 512], {"block": 2}), ([2, 1024], {"block": 3})]
    assert by_size(plan) == {(2, 1024): [{"block": 1}, {"block": 3}], (4, 512): [{"block": 2}]}
    assert list(by_size(plan)) == [(2, 1024), (4, 512)]


def test_bursts_cover_every_config_once() -> None:
    configs = [{"block": block} for block in range(5)]
    assert list(bursts(configs, 2)) == [configs[0:2], configs[2:4], configs[4:5]]


def test_style_filter_matches_part_of_the_style() -> None:
    assert selected({"style": "whole/pointer"}, "whole", {})
    assert not selected({"style": "looped/pointer"}, "whole", {})


def test_field_filter_needs_an_exact_value() -> None:
    config = {"style": "whole", "num_stages": 10}
    assert selected(config, "", {"num_stages": "10"})
    assert not selected(config, "", {"num_stages": "1"})


def test_field_filter_ignores_fields_the_target_does_not_have() -> None:
    assert selected({"style": "whole", "num_warps": 8}, "", {"num_stages": "5"})
