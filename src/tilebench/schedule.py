"""Count how many global loads a warp can have in flight, read from SASS.

A memory-bound kernel needs many loads in flight at once, and that can be read off the
schedule without a profiler. A load stays in flight until the first instruction that names
one of its registers. The counts are upper bounds: loads that share a scoreboard are waited
for together, which this does not see.

Only LDG counts, so the shared memory pipeline of looped pointer kernels is not covered.
"""

import re
from dataclasses import dataclass

INSTRUCTION = re.compile(
    r"^\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)([^;\n]*);", re.MULTILINE
)
REGISTER = re.compile(r"(?<![A-Za-z0-9_])R(\d+)")
REGISTERS_BY_WIDTH = {".128": 4, ".64": 2}


@dataclass(frozen=True)
class Instruction:
    name: str
    operands: str

    @property
    def base(self) -> str:
        return self.name.split(".")[0]

    @property
    def registers(self) -> set[int]:
        return {int(number) for number in REGISTER.findall(self.operands)}

    @property
    def load_target(self) -> set[int]:
        """Return the registers a load writes, which its operands only name the first of."""
        first = REGISTER.search(self.operands)
        if first is None:
            raise ValueError(f"no destination register in {self.name} {self.operands}")
        widths = REGISTERS_BY_WIDTH.items()
        count = next((size for suffix, size in widths if self.name.endswith(suffix)), 1)
        return set(range(int(first.group(1)), int(first.group(1)) + count))


@dataclass(frozen=True)
class LoadSchedule:
    branch_regions: int
    first_peak: int
    last_peak: int
    spills: int


def parse(sass: str) -> list[Instruction]:
    return [Instruction(*match) for match in INSTRUCTION.findall(sass)]


def hot_path(instructions: list[Instruction]) -> list[Instruction]:
    """Drop the slow paths after the last EXIT, and take every branch as not taken."""
    last_exit = max(index for index, item in enumerate(instructions) if item.base == "EXIT")
    return instructions[: last_exit + 1]


def loads_in_flight(instructions: list[Instruction]) -> list[list[int]]:
    """Return the count after each load issues, for each stretch between barriers."""
    stretches: list[list[int]] = [[]]
    waiting: dict[int, int] = {}
    for number, item in enumerate(hot_path(instructions)):
        is_load = item.base == "LDG"
        if item.base == "BAR":
            stretches.append([])
        # A load's own target is written, not read, so it does not end an earlier load.
        read = item.registers - item.load_target if is_load else item.registers
        finished = {waiting[register] for register in read if register in waiting}
        waiting = {reg: load for reg, load in waiting.items() if load not in finished}
        if is_load:
            waiting.update(dict.fromkeys(item.load_target, number))
            stretches[-1].append(len(set(waiting.values())))
    return [stretch for stretch in stretches if stretch]


def read_schedule(sass: str) -> LoadSchedule:
    """Summarize a kernel, whose first and last stretches are what a layer norm cares about."""
    instructions = parse(sass)
    bases = [item.base for item in hot_path(instructions)]
    stretches = loads_in_flight(instructions) or [[0]]
    return LoadSchedule(
        branch_regions=bases.count("BSSY"),
        first_peak=max(stretches[0]),
        last_peak=max(stretches[-1]),
        spills=bases.count("LDL"),
    )
