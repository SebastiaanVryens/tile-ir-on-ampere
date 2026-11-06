from collections.abc import Iterable

import pytest

from tilebench import system


def fake_temperatures(monkeypatch: pytest.MonkeyPatch, readings: Iterable[int]) -> list[float]:
    """Replace the GPU readings and return the list that records each sleep."""
    values = iter(readings)
    sleeps: list[float] = []
    monkeypatch.setattr(system, "gpu_temperature", lambda: next(values))
    monkeypatch.setattr(system.time, "sleep", sleeps.append)
    return sleeps


def test_a_cool_gpu_does_not_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps = fake_temperatures(monkeypatch, [60])
    assert system.wait_until_cool(70) == 60
    assert sleeps == []


def test_the_limit_itself_counts_as_cool(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_temperatures(monkeypatch, [70])
    assert system.wait_until_cool(70) == 70


def test_a_hot_gpu_waits_until_it_reaches_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps = fake_temperatures(monkeypatch, [85, 78, 69])
    assert system.wait_until_cool(70) == 69
    assert len(sleeps) == 2


def test_a_gpu_that_never_cools_returns_its_real_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_temperatures(monkeypatch, [85, 84])
    assert system.wait_until_cool(70, timeout_s=0) == 85
