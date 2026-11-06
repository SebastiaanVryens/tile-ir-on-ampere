from tilebench.timing import LAUNCHES_PER_BURST, burst_length


def test_fewer_repeats_fit_more_kernels_in_a_burst() -> None:
    assert burst_length(20) == LAUNCHES_PER_BURST // 20
    assert burst_length(100) == LAUNCHES_PER_BURST // 100


def test_a_burst_always_holds_at_least_one_kernel() -> None:
    assert burst_length(LAUNCHES_PER_BURST * 2) == 1
