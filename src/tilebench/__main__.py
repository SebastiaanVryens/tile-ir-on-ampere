"""Run the benchmarks and print a comparison: `python -m tilebench`."""

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tilebench.kernels import WORKLOAD_NAMES
from tilebench.report import best_rows, format_table, gpu_state_note
from tilebench.system import describe_environment
from tilebench.targets import TARGETS, TILEIR_SITE, Target, get_target, worker_env


def run_worker(target: Target, workload: str) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = Path(tmp_dir) / "records.json"
        command = [
            sys.executable,
            "-m",
            "tilebench.worker",
            "--target",
            target.name,
            "--workload",
            workload,
            "--out",
            str(out),
        ]
        subprocess.run(command, env=worker_env(target), check=True)
        return json.loads(out.read_text())


def run_rounds(
    targets: Sequence[Target], workloads: Sequence[str], rounds: int
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for round_index in range(rounds):
        # Rotating the order stops one target always running on a cooler GPU.
        offset = round_index % len(targets)
        order = [*targets[offset:], *targets[:offset]]
        for workload in workloads:
            for target in order:
                print(f"round {round_index + 1}/{rounds}: {workload} on {target.name}", flush=True)
                for record in run_worker(target, workload):
                    records.append(record | {"round": round_index})
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--targets", nargs="+", default=[target.name for target in TARGETS])
    parser.add_argument("--workloads", nargs="+", default=list(WORKLOAD_NAMES))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if not TILEIR_SITE.is_dir():
        print(f"error: {TILEIR_SITE} not found; run scripts/setup_wsl.sh first", file=sys.stderr)
        return 2

    targets = [get_target(name) for name in args.targets]
    records = run_rounds(targets, args.workloads, args.rounds)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = args.out_dir / f"{stamp}.json"
    path.write_text(json.dumps({"environment": describe_environment(), "records": records}))

    print(format_table(best_rows(records)))
    print(gpu_state_note(records))
    print(f"\nRaw results: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
