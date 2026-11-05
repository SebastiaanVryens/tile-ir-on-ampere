from tilebench.kernels.layernorm import loop_blocks, triton_impl, whole_row_block


def test_whole_row_block_is_the_next_power_of_two() -> None:
    assert whole_row_block(1000) == 1024
    assert whole_row_block(1024) == 1024
    assert whole_row_block(1025) == 2048


def test_short_rows_do_not_get_oversized_loop_blocks() -> None:
    assert loop_blocks(8) == (8,)
    assert loop_blocks(1000) == (1024,)


def test_long_rows_try_every_loop_block() -> None:
    assert loop_blocks(65536) == (1024, 4096)


def test_unaligned_rows_skip_descriptors() -> None:
    styles = {config["style"] for config in triton_impl.configs("cuda", (2, 1025))}
    assert styles == {"whole/pointer", "looped/pointer"}


def test_aligned_rows_try_every_style() -> None:
    styles = {config["style"] for config in triton_impl.configs("cuda", (2, 1024))}
    assert styles == set(triton_impl.KERNELS)


def test_each_backend_sweeps_its_own_knob() -> None:
    assert all("num_warps" in config for config in triton_impl.configs("cuda", (2, 1024)))
    assert all("occupancy" in config for config in triton_impl.configs("tileir", (2, 1024)))
