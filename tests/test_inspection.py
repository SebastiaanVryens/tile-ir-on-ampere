"""Inspection must report what the compilers really built, checked against known facts."""

import shutil
from pathlib import Path
from typing import Any

import pytest

from tilebench.inspection import INSPECTABLE, OPCODE_COLUMNS, format_inspection
from tilebench.targets import get_target, run_worker

needs_gpu = pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="needs a GPU")


def inspect(target: str, out_dir: Path, style: str = "whole") -> list[dict[str, Any]]:
    options = ("--inspect-size", "2x32768", "--style", style, "--artifacts", str(out_dir))
    return run_worker(get_target(target), "layernorm", *options)


def test_table_shows_the_variant_and_its_resources() -> None:
    record = {
        "target": "triton-tileir",
        "config": {"style": "whole/pointer", "block": 32768},
        "threads": 128,
        "registers": 255,
        "stack_bytes": 400,
        "shared_bytes": 16,
        "warps_per_sm": 8,
        "opcodes": {opcode: 0 for opcode in OPCODE_COLUMNS},
        "instructions": 900,
    }
    table = format_inspection([record])
    assert "triton-tileir/whole/pointer" in table
    assert "block=32768" in table
    assert "255" in table
    assert "local memory" in table


@needs_gpu
@pytest.mark.parametrize("target", INSPECTABLE)
def test_every_target_reports_a_runnable_kernel(target: str, tmp_path: Path) -> None:
    records = inspect(target, tmp_path)
    assert records
    for record in records:
        assert record["threads"] % 32 == 0
        assert record["registers"] > 0
        assert record["blocks_per_sm"] >= 1
    assert list(tmp_path.glob("*/*/kernel.sass"))


@needs_gpu
def test_ptx_threads_match_the_requested_warps(tmp_path: Path) -> None:
    for record in inspect("triton-ptx", tmp_path, "whole/pointer"):
        assert record["threads"] == record["config"]["num_warps"] * 32


@needs_gpu
def test_tile_ir_stops_at_eight_warps(tmp_path: Path) -> None:
    threads = {
        record["config"]["num_warps"]: record["threads"]
        for record in inspect("triton-tileir", tmp_path, "whole/pointer")
    }
    assert threads == {4: 128, 8: 256}


@needs_gpu
def test_tile_ir_text_is_saved_for_both_tile_ir_paths(tmp_path: Path) -> None:
    inspect("triton-tileir", tmp_path / "triton", "whole/pointer")
    inspect("cutile", tmp_path / "cutile")
    assert list((tmp_path / "triton").glob("*/*/kernel.tileir"))
    assert list((tmp_path / "cutile").glob("*/*/kernel.tileir"))
