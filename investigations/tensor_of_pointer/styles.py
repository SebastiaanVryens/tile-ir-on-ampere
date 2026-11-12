"""Pointer against descriptor loads on both Triton backends, from saved benchmark runs.

Usage: python investigations/tensor_of_pointer/styles.py results/RUN.json [...] [--workloads NAME ...]

Reads the JSON files that `python -m tilebench` writes and needs no GPU. The extra time is
how much longer the descriptor kernel takes than the pointer kernel, so a positive value
means pointer loads are faster. Two views:
  best    each style's reported winner, pooled over the three rounds
  pairs   tuning configurations that differ only in the load style, from the same run:
          the median extra time, and how many pairs pointer loads win, descriptors win,
          and tie within 2%
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

BACKENDS = {"triton-tileir": "Tile IR", "triton-ptx": "PTX"}
TIE_PCT = 2.0


def kernel_and_style(style: str) -> tuple[str, str]:
    """Split a style such as 'whole/pointer' into the kernel and the load style."""
    kernel, _, load = style.rpartition("/")
    return kernel, load


def extra_time(pointer_ms: float, descriptor_ms: float) -> float:
    return (descriptor_ms / pointer_ms - 1) * 100


def best(records: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, float]]:
    pooled: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for record in records:
        kernel, load = kernel_and_style(record["config"]["style"])
        pooled[(record["workload"], kernel, tuple(record["size"]), record["target"], load)] += record["gpu_ms"]
    rows: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for (workload, kernel, size, target, load), samples in pooled.items():
        rows[(workload, kernel, size, target)][load] = statistics.median(samples)
    return rows


def pairs(records: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[float]]:
    tuned: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for record in records:
        config = dict(record["config"])
        kernel, load = kernel_and_style(config.pop("style"))
        key = (record["workload"], kernel, tuple(record["size"]), record["target"],
               json.dumps(config, sort_keys=True))
        tuned[key][load] = statistics.median(record["gpu_ms"])
    deltas: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for key, loads in tuned.items():
        if {"pointer", "descriptor"} <= loads.keys():
            deltas[key[:4]].append(extra_time(loads["pointer"], loads["descriptor"]))
    return deltas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--workloads", nargs="*", help="only these workloads, such as layernorm")
    args = parser.parse_args()
    for path in args.runs:
        data = json.loads(path.read_text())

        def wanted(record: dict[str, Any]) -> bool:
            workload_ok = not args.workloads or record["workload"] in args.workloads
            return workload_ok and record["target"] in BACKENDS

        rounds = best([r for r in data["records"] if wanted(r)])
        tuning = pairs([r for r in data.get("tuning", []) if wanted(r)])
        cases = sorted({key[:3] for key in rounds} | {key[:3] for key in tuning})
        print(f"\n{path.name}: extra time of descriptor loads")
        print(f"  {'kernel':<22}{'size':<16}" + "".join(f"{'best, ' + label:>16}" for label in BACKENDS.values())
              + "".join(f"{'pairs, ' + label:>30}" for label in BACKENDS.values()))
        for workload, kernel, size in cases:
            cells = []
            for target in BACKENDS:
                loads = rounds.get((workload, kernel, size, target), {})
                if {"pointer", "descriptor"} <= loads.keys():
                    cells.append(f"{extra_time(loads['pointer'], loads['descriptor']):>+15.1f}%")
                else:
                    cells.append(f"{'-':>16}")
            for target in BACKENDS:
                deltas = tuning.get((workload, kernel, size, target), [])
                if not deltas:
                    cells.append(f"{'-':>30}")
                    continue
                pointer = sum(d > TIE_PCT for d in deltas)
                descriptor = sum(d < -TIE_PCT for d in deltas)
                ties = len(deltas) - pointer - descriptor
                cells.append(f"{statistics.median(deltas):>+12.1f}%  {pointer:>4} /{descriptor:>4} /{ties:>4}")
            label = f"{workload} {kernel}".strip()
            print(f"  {label:<22}{'x'.join(map(str, size)):<16}{''.join(cells)}")


if __name__ == "__main__":
    main()
