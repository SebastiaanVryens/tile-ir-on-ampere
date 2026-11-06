"""Assemble Tile IR text into bytecode and compile it, so that single IR lines can be edited.

No assembler ships as a tool. Triton's Tile IR build exposes one through its extension, so
these scripts need that build on the path, which run.sh arranges.
"""

import os
import subprocess
import sysconfig
import tempfile
from pathlib import Path

from triton._C.libtriton import ir, tileir

from tilebench.cubin import Kernel, read_kernel
from tilebench.targets import TILEIR_SITE

# cuTile uses the pip build and Triton its own, so step one was to compare the two.
COMPILER_HOMES = {
    "pip": Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13",
    "bundled": TILEIR_SITE / "triton" / "backends" / "nvidia" / "tileir_cuda",
}


def wrapped(text: str) -> str:
    """cuTile's listing is a bare entry, while the parser wants a complete module."""
    if "cuda_tile.module" in text:
        return text
    return f"module {{\n  cuda_tile.module @cuda_tile_module {{\n{text}\n  }}\n}}\n"


def assemble(text: str) -> bytes:
    context = ir.context()
    ir.load_dialects(context)
    tileir.load_dialects(context)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "kernel.mlir"
        path.write_text(wrapped(text))
        return bytes(tileir.write_bytecode(ir.parse_mlir_module(str(path), context)))


def compile_bytecode(bytecode: bytes, compiler: str = "bundled") -> Kernel:
    home = COMPILER_HOMES[compiler]
    with tempfile.TemporaryDirectory() as folder:
        source, target = Path(folder) / "kernel.bytecode", Path(folder) / "kernel.cubin"
        source.write_bytes(bytecode)
        command = [
            str(home / "bin" / "tileiras"),
            "--gpu-name=sm_86",
            "--opt-level=3",
            "--lineinfo",
            str(source),
            "-o",
            str(target),
        ]
        subprocess.run(
            command,
            env={**os.environ, "CUDA_HOME": str(home)},
            check=True,
            capture_output=True,
            timeout=600,
        )
        return read_kernel(target.read_bytes())


def instruction_stream(sass: str) -> list[str]:
    """Keep the encoded instructions only, so that a kernel name cannot fail a comparison."""
    lines = (line.strip() for line in sass.splitlines())
    return [line for line in lines if line.startswith("/*") and "/* 0x" in line]
