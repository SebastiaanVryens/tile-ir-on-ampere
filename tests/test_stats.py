import pytest

from tilebench.stats import summarize


def test_summarize_odd_sample_count() -> None:
    summary = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary.median == 3.0
    assert summary.q1 == 2.0
    assert summary.q3 == 4.0
    assert summary.iqr == 2.0


def test_summarize_ignores_input_order() -> None:
    assert summarize([5.0, 1.0, 3.0]) == summarize([1.0, 3.0, 5.0])


def test_median_resists_one_outlier() -> None:
    assert summarize([1.0, 1.0, 1.0, 1.0, 1000.0]).median == pytest.approx(1.0)
