from tilebench.kernels.gemm import (
    LATENCY_HINTS,
    OCCUPANCIES,
    TUTORIAL_TILES,
    cutile_impl,
    flops,
    triton_impl,
)

ALIGNED = (64, 64, 64)


def tile_of(config: dict[str, object]) -> tuple[object, ...]:
    return config["tile_m"], config["tile_n"], config["tile_k"]


def test_flops_count_a_multiply_and_an_add_per_term() -> None:
    assert flops((2, 3, 4)) == 48


def test_aligned_shapes_try_both_load_styles() -> None:
    styles = {config["style"] for config in triton_impl.configs("cuda", ALIGNED)}
    assert styles == set(triton_impl.KERNELS)


def test_unaligned_rows_of_a_or_b_skip_descriptors() -> None:
    for size in ((64, 64, 65), (64, 65, 64)):
        styles = {config["style"] for config in triton_impl.configs("cuda", size)}
        assert styles == {"pointer"}


def test_m_does_not_need_alignment() -> None:
    styles = {config["style"] for config in triton_impl.configs("cuda", (65, 64, 64))}
    assert styles == set(triton_impl.KERNELS)


def test_every_path_tries_every_tutorial_tile() -> None:
    expected = {tile[:3] for tile in TUTORIAL_TILES}
    assert {tile_of(config) for config in cutile_impl.configs("cutile", ALIGNED)} == expected
    for backend in ("cuda", "tileir"):
        assert {tile_of(config) for config in triton_impl.configs(backend, ALIGNED)} == expected


def test_triton_uses_the_tutorial_warps() -> None:
    warps = {tile[:3]: tile[3] for tile in TUTORIAL_TILES}
    for backend in ("cuda", "tileir"):
        for config in triton_impl.configs(backend, ALIGNED):
            assert config["num_warps"] == warps[tile_of(config)]


def test_ptx_pipelines_fit_in_shared_memory() -> None:
    for config in triton_impl.configs("cuda", ALIGNED):
        size = triton_impl.pipeline_bytes(*tile_of(config), config["num_stages"])
        assert size <= triton_impl.SHARED_LIMIT_BYTES
        assert "occupancy" not in config


def test_the_largest_tile_keeps_only_the_stages_that_fit() -> None:
    stages = {
        config["num_stages"]
        for config in triton_impl.configs("cuda", ALIGNED)
        if tile_of(config) == (128, 256, 64)
    }
    assert stages == {2, 3}


def test_both_tile_ir_paths_sweep_the_same_hints_and_occupancies() -> None:
    triton_configs = triton_impl.configs("tileir", ALIGNED)
    cutile_configs = cutile_impl.configs("cutile", ALIGNED)
    assert {config["num_stages"] for config in triton_configs} == set(LATENCY_HINTS)
    assert {config["latency"] for config in cutile_configs} == {None, *LATENCY_HINTS}
    assert {config["occupancy"] for config in triton_configs} == set(OCCUPANCIES)
    assert {config["occupancy"] for config in cutile_configs} == set(OCCUPANCIES)
