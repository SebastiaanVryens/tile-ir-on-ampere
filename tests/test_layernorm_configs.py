from tilebench.kernels.layernorm import (
    LATENCY_HINTS,
    cutile_impl,
    loop_blocks,
    triton_impl,
    whole_row_block,
)


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


def test_ptx_sweeps_warps_only() -> None:
    configs = triton_impl.configs("cuda", (2, 1024))
    assert {config["num_warps"] for config in configs} == {4, 8, 16, 32}
    assert not any("occupancy" in config or "num_stages" in config for config in configs)


def test_tile_ir_sweeps_warps_and_occupancy() -> None:
    configs = triton_impl.configs("tileir", (2, 1024))
    assert {config["num_warps"] for config in configs} == {4, 8}
    assert {config["occupancy"] for config in configs} == {1, 2, 4, 8}


def test_cutile_tries_the_compiler_choice_and_eight_warps() -> None:
    configs = cutile_impl.configs("cutile", (2, 1024))
    assert {config["num_worker_warps"] for config in configs} == {None, 8}


def test_both_tile_ir_paths_sweep_the_same_latency_hints() -> None:
    triton_hints = {config["num_stages"] for config in triton_impl.configs("tileir", (2, 1024))}
    cutile_hints = {config["latency"] for config in cutile_impl.configs("cutile", (2, 1024))}
    assert triton_hints == set(LATENCY_HINTS)
    assert cutile_hints == {None, *LATENCY_HINTS}


def test_every_strategy_gets_the_latency_sweep() -> None:
    configs = triton_impl.configs("tileir", (2, 1024))
    for style in triton_impl.KERNELS:
        hints = {config["num_stages"] for config in configs if config["style"] == style}
        assert hints == set(LATENCY_HINTS)
