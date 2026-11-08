from types import SimpleNamespace
from typing import Any

import pytest

from tilebench import worker
from tilebench.kernels import gemm, layernorm
from tilebench.targets import get_target
from tilebench.worker import by_size, require_backend, selected, start_limit, timed_burst


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeTimer:
    """Advances the clock by a fixed kernel time instead of running anything."""

    def __init__(self, clock: FakeClock, kernel_s: float) -> None:
        self.clock = clock
        self.kernel_s = kernel_s

    def gpu_ms(self, launch: Any, repeats: int) -> list[float]:
        self.clock.now += self.kernel_s
        return [self.kernel_s * 1e3 / repeats] * repeats


def burst_sizes(monkeypatch: pytest.MonkeyPatch, kernel_s: float, configs: int) -> list[int]:
    """Return how many configs land in each burst."""
    clock = FakeClock()
    monkeypatch.setattr(worker.time, "perf_counter", clock)
    impl: Any = SimpleNamespace(prepare=lambda inputs, config: SimpleNamespace(launch=None))
    timer: Any = FakeTimer(clock, kernel_s)
    pending: list[dict[str, Any]] = [{"block": block} for block in range(configs)]
    sizes = []
    while pending:
        timings = timed_burst(timer, impl, (), pending, 100)
        sizes.append(len(timings))
        pending = pending[len(timings) :]
    return sizes


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


def test_short_kernels_share_bursts_up_to_the_launch_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert burst_sizes(monkeypatch, kernel_s=0.05, configs=14) == [6, 6, 2]


def test_long_kernels_get_a_burst_each(monkeypatch: pytest.MonkeyPatch) -> None:
    assert burst_sizes(monkeypatch, kernel_s=4.0, configs=3) == [1, 1, 1]


def test_gemm_starts_cooler_than_requested() -> None:
    assert start_limit(gemm.WORKLOAD, 70) == gemm.START_TEMP_C


def test_other_workloads_keep_the_requested_start() -> None:
    assert start_limit(layernorm.WORKLOAD, 70) == 70


def test_a_cooler_request_wins() -> None:
    assert start_limit(gemm.WORKLOAD, 50) == 50


def test_style_filter_matches_part_of_the_style() -> None:
    assert selected({"style": "whole/pointer"}, "whole", {})
    assert not selected({"style": "looped/pointer"}, "whole", {})


def test_field_filter_needs_an_exact_value() -> None:
    config = {"style": "whole", "num_stages": 10}
    assert selected(config, "", {"num_stages": "10"})
    assert not selected(config, "", {"num_stages": "1"})


def test_field_filter_ignores_fields_the_target_does_not_have() -> None:
    assert selected({"style": "whole", "num_warps": 8}, "", {"num_stages": "5"})
