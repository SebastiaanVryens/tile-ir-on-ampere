from typing import Any

import pytest

from tilebench.report import (
    best_rows,
    config_label,
    format_table,
    gpu_state_note,
    variant_name,
)


def record(
    target: str = "triton-tileir",
    config: dict[str, Any] | None = None,
    gpu_ms: list[float] | None = None,
    size: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "workload": "vector_add",
        "target": target,
        "size": size or [1024],
        "config": {"style": "pointer", "block": 1024} if config is None else config,
        "gpu_ms": gpu_ms or [1.0, 1.0, 1.0],
        "work": 1e9,
        "rate_unit": "GB/s",
        "rate_scale": 1e9,
    }


def test_load_styles_stay_separate_variants() -> None:
    assert variant_name(record()) == "triton-tileir/pointer"
    assert variant_name(record(target="cutile", config={"tile": 1024})) == "cutile"


def test_style_is_not_repeated_in_the_config_label() -> None:
    assert config_label({"style": "pointer", "block": 1024}) == "block=1024"
    assert config_label({}) == "-"


def test_fastest_config_wins_per_variant_and_size() -> None:
    slow = record(config={"style": "pointer", "block": 1024}, gpu_ms=[2.0, 2.0, 2.0])
    fast = record(config={"style": "pointer", "block": 4096}, gpu_ms=[1.0, 1.0, 1.0])
    (row,) = best_rows([slow, fast])
    assert row.config == "block=4096"
    assert row.gpu_ms.median == 1.0


def test_load_styles_are_not_collapsed_by_best_config() -> None:
    pointer = record(config={"style": "pointer", "block": 1024})
    descriptor = record(config={"style": "descriptor", "block": 1024})
    assert len(best_rows([pointer, descriptor])) == 2


def test_rounds_are_pooled_and_drift_is_reported() -> None:
    rounds = [record(gpu_ms=[1.0, 1.0, 1.0]), record(gpu_ms=[1.2, 1.2, 1.2])]
    (row,) = best_rows(rounds)
    assert row.gpu_ms.median == pytest.approx(1.1)
    assert row.drift_pct == pytest.approx(0.2 / 1.1 * 100)


def test_rate_comes_from_the_median() -> None:
    (row,) = best_rows([record(gpu_ms=[1.0, 2.0, 100.0])])
    assert row.rate == pytest.approx(1e9 / 2e-3 / 1e9)


def test_gpu_state_note_reports_the_full_range() -> None:
    cool = {"gpu_state": {"clocks.sm": "1995", "temperature.gpu": "63"}}
    hot = {"gpu_state": {"clocks.sm": "1702", "temperature.gpu": "87"}}
    note = gpu_state_note([cool, hot])
    assert "1702-1995 MHz" in note
    assert "63-87 C" in note


def test_table_has_one_section_per_size_and_warns_about_the_timer() -> None:
    table = format_table(best_rows([record(size=[1024]), record(size=[2048])]))
    assert "size=1024" in table
    assert "size=2048" in table
    assert "about 1 us" in table
