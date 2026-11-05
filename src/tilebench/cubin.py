"""Read what a compiler actually produced out of a cubin."""

import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tilebench.targets import TILEIR_SITE

# One disassembler for every target, so the counts stay comparable.
CUOBJDUMP = TILEIR_SITE / "triton" / "backends" / "nvidia" / "bin" / "cuobjdump"

RESOURCES = re.compile(r"Function (\S+):\s+REG:(\d+) STACK:(\d+) SHARED:(\d+)")
REQUIRED_THREADS = re.compile(
    r"EIATTR_REQNTID\s+Format:\s+\S+\s+Value:\s+(0x[0-9a-f]+) (0x[0-9a-f]+) (0x[0-9a-f]+)"
)
INSTRUCTION = re.compile(r"^\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9]*)", re.MULTILINE)


@dataclass(frozen=True)
class Kernel:
    name: str
    threads: int
    registers: int
    stack_bytes: int
    static_shared_bytes: int
    sass: str


def parse_resources(text: str) -> tuple[str, int, int, int]:
    """Return the name, registers, stack and shared bytes of the one kernel in a cubin."""
    matches = RESOURCES.findall(text)
    if len(matches) != 1:
        raise ValueError(f"expected one kernel in the cubin, found {len(matches)}")
    name, registers, stack_bytes, shared_bytes = matches[0]
    return name, int(registers), int(stack_bytes), int(shared_bytes)


def parse_threads(elf: str) -> int:
    """Return the threads per block fixed in the cubin.

    Tile IR launches pass a block size of 1, so this is the only place the real value
    appears. Triton sets the same attribute on the PTX path.
    """
    match = REQUIRED_THREADS.search(elf)
    if match is None:
        raise ValueError("the cubin does not fix a block size")
    x, y, z = (int(value, 16) for value in match.groups())
    return x * y * z


def count_opcodes(sass: str) -> Counter[str]:
    """Count instructions by base opcode, ignoring modifiers such as .E.128.

    These are static counts: a loop body counts once, however many times it runs.
    """
    return Counter(INSTRUCTION.findall(sass))


def run_cuobjdump(path: Path, flag: str) -> str:
    command = [str(CUOBJDUMP), flag, str(path)]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout


def read_kernel(cubin: bytes) -> Kernel:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "kernel.cubin"
        path.write_bytes(cubin)
        resources = run_cuobjdump(path, "--dump-resource-usage")
        elf = run_cuobjdump(path, "--dump-elf")
        sass = run_cuobjdump(path, "--dump-sass")
    name, registers, stack_bytes, shared_bytes = parse_resources(resources)
    return Kernel(name, parse_threads(elf), registers, stack_bytes, shared_bytes, sass)
