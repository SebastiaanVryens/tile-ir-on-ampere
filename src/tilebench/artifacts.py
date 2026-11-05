"""Collect what each compiler produced for one kernel, so the paths can be compared."""

import io
import subprocess
import sysconfig
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from cuda.bindings import driver

from tilebench.cubin import Kernel, count_opcodes, read_kernel
from tilebench.targets import Target
from tilebench.workload import Config, Prepared

TILEIRDISASM = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "bin" / "tileirdisasm"

REPORTED_OPCODES = ("LDG", "LDGSTS", "LDS", "LDL", "STL", "BAR", "HMMA")


@dataclass(frozen=True)
class Artifacts:
    cubin: bytes
    stages: dict[str, str]
    dynamic_shared_bytes: int


def from_triton(compiled: Any) -> Artifacts:
    stages = {stage: text for stage, text in compiled.asm.items() if isinstance(text, str)}
    # On Tile IR this field repeats the shared memory already fixed in the cubin, so
    # adding it would count that memory twice. Only the PTX path requests more at launch.
    dynamic = compiled.metadata.shared if compiled.metadata.target.backend == "cuda" else 0
    return Artifacts(compiled.asm["cubin"], stages, dynamic)


def disassemble_tile_ir(bytecode: bytes) -> str:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "kernel.tileirbc"
        path.write_bytes(bytecode)
        command = [str(TILEIRDISASM), str(path)]
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout


def from_cutile(kernel: Any, args: tuple[Any, ...]) -> Artifacts:
    # Imported here so that the Triton targets never load cuTile.
    from cuda.tile.compilation import CallingConvention, KernelSignature, export_kernel

    major, minor = torch.cuda.get_device_capability()
    # With this convention the exported cubin is byte-identical to the one ct.launch runs.
    signature = KernelSignature.from_kernel_args(kernel, args, CallingConvention.cutile_python_v1())

    def export(output_format: Any) -> bytes:
        buffer = io.BytesIO()
        export_kernel(
            kernel,
            [signature],
            buffer,
            gpu_code=f"sm_{major}{minor}",
            output_format=output_format,
        )
        return buffer.getvalue()

    stages = {"tileir": disassemble_tile_ir(export("tileir_bytecode"))}
    return Artifacts(export("cubin"), stages, 0)


def collect(target: Target, prepared: Prepared) -> Artifacts:
    if prepared.tile_kernel is not None:
        return from_cutile(*prepared.tile_kernel)
    if target.impl == "triton_impl":
        return from_triton(prepared.launch())
    raise ValueError(f"{target.name} does not produce a kernel to inspect")


def checked(result: tuple[Any, ...]) -> Any:
    error, *values = result
    if error != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {error}")
    return values[0] if values else None


def resident_blocks(artifacts: Artifacts, kernel: Kernel) -> int:
    """Ask the driver how many blocks of this kernel fit on one SM at once."""
    module = checked(driver.cuModuleLoadData(artifacts.cubin))
    try:
        function = checked(driver.cuModuleGetFunction(module, kernel.name.encode()))
        limit = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        checked(driver.cuFuncSetAttribute(function, limit, artifacts.dynamic_shared_bytes))
        return checked(
            driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                function, kernel.threads, artifacts.dynamic_shared_bytes
            )
        )
    finally:
        checked(driver.cuModuleUnload(module))


def config_slug(config: Config) -> str:
    return "_".join(f"{key}-{value}" for key, value in config.items()).replace("/", "-")


def save(artifacts: Artifacts, kernel: Kernel, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.cubin").write_bytes(artifacts.cubin)
    (directory / "kernel.sass").write_text(kernel.sass)
    for stage, text in artifacts.stages.items():
        (directory / f"kernel.{stage}").write_text(text)


def summarize(artifacts: Artifacts, kernel: Kernel) -> dict[str, Any]:
    blocks = resident_blocks(artifacts, kernel)
    opcodes = count_opcodes(kernel.sass)
    return {
        "threads": kernel.threads,
        "registers": kernel.registers,
        "stack_bytes": kernel.stack_bytes,
        "shared_bytes": kernel.static_shared_bytes + artifacts.dynamic_shared_bytes,
        "blocks_per_sm": blocks,
        "warps_per_sm": blocks * kernel.threads // 32,
        "opcodes": {opcode: opcodes[opcode] for opcode in REPORTED_OPCODES},
        "instructions": opcodes.total() - opcodes["NOP"],
    }


def inspect_build(target: Target, prepared: Prepared, directory: Path) -> dict[str, Any]:
    artifacts = collect(target, prepared)
    kernel = read_kernel(artifacts.cubin)
    save(artifacts, kernel, directory)
    return summarize(artifacts, kernel)
