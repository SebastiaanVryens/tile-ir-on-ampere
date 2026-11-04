from tilebench.targets import TILEIR_SITE, get_target, worker_env


def test_inherited_backend_settings_are_removed() -> None:
    base = {
        "PATH": "/bin",
        "PYTHONPATH": "/stale",
        "ENABLE_TILE": "1",
        "TRITON_DEFAULT_BACKEND": "tileir",
        "TRITON_TILEIR_RUNTIME_FALLBACK": "1",
    }
    assert worker_env(get_target("torch"), base) == {"PATH": "/bin"}


def test_ptx_target_uses_the_wheel_without_enabling_tile_ir() -> None:
    env = worker_env(get_target("triton-ptx"), {"ENABLE_TILE": "1"})
    assert env == {"PYTHONPATH": str(TILEIR_SITE)}


def test_tileir_target_enables_the_backend_with_fallback_off() -> None:
    env = worker_env(get_target("triton-tileir"), {"TRITON_TILEIR_RUNTIME_FALLBACK": "1"})
    assert env == {"PYTHONPATH": str(TILEIR_SITE), "ENABLE_TILE": "1"}
