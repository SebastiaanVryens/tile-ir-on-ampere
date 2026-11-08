import pytest

from tilebench.timing import LAUNCHES_PER_BURST, MAX_BURST_S, burst_has_room


def kernels_per_burst(repeats: int, kernel_s: float) -> int:
    count = 0
    while burst_has_room(count * repeats, count * kernel_s, kernel_s if count else 0.0, repeats):
        count += 1
    return count


def test_a_burst_always_holds_at_least_one_kernel() -> None:
    assert burst_has_room(0, 0.0, 0.0, LAUNCHES_PER_BURST * 2)
    assert kernels_per_burst(100, MAX_BURST_S * 2) == 1


@pytest.mark.parametrize("repeats", [20, 100])
def test_short_kernels_keep_the_launch_limit(repeats: int) -> None:
    assert kernels_per_burst(repeats, 0.001) == LAUNCHES_PER_BURST // repeats


def test_long_kernels_stop_before_the_time_limit() -> None:
    assert kernels_per_burst(100, 4.0) == 1
    assert kernels_per_burst(20, 0.8) == 6
