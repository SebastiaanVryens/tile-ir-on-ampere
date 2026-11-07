from tilebench.kernels.gemm import TILES, cutile_impl, flops, triton_impl


def tile_of(config: dict[str, object]) -> tuple[object, ...]:
    return config["tile_m"], config["tile_n"], config["tile_k"]


def test_flops_count_a_multiply_and_an_add_per_term() -> None:
    assert flops((2, 3, 4)) == 48


def test_aligned_shapes_try_both_load_styles() -> None:
    styles = {config["style"] for config in triton_impl.configs("cuda", (64, 64, 64))}
    assert styles == set(triton_impl.KERNELS)


def test_unaligned_rows_of_a_or_b_skip_descriptors() -> None:
    for size in ((64, 64, 65), (64, 65, 64)):
        styles = {config["style"] for config in triton_impl.configs("cuda", size)}
        assert styles == {"pointer"}


def test_m_does_not_need_alignment() -> None:
    styles = {config["style"] for config in triton_impl.configs("cuda", (65, 64, 64))}
    assert styles == set(triton_impl.KERNELS)


def test_every_path_tries_the_same_tiles() -> None:
    expected = {tile_of(tile) for tile in TILES}
    size = (64, 64, 64)
    assert {tile_of(config) for config in cutile_impl.configs("cutile", size)} == expected
    for backend in ("cuda", "tileir"):
        assert {tile_of(config) for config in triton_impl.configs(backend, size)} == expected
