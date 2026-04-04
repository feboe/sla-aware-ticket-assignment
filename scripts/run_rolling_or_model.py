"""CLI entrypoint for running the rolling-horizon CP-SAT ticket-assignment benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.or_rolling import (
    DEFAULT_ROLLING_TIME_LIMIT_SEC,
    run_rolling_or_model_from_csv,
    write_rolling_or_outputs,
)

KNOWN_SOLVER_STATUSES = ("OPTIMAL", "FEASIBLE", "UNKNOWN")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the rolling OR replay."""

    parser = argparse.ArgumentParser(
        description="Replay the dataset with a rolling CP-SAT ticket-assignment policy."
    )
    parser.add_argument(
        "--tickets",
        default="data/tickets.csv",
        help="Path to the ticket-demand CSV.",
    )
    parser.add_argument(
        "--agents",
        default="data/agents.csv",
        help="Path to the agent-supply CSV.",
    )
    parser.add_argument(
        "--schedule-out",
        default="results/rolling_or_schedule.csv",
        help="Path to the generated rolling OR schedule CSV.",
    )
    parser.add_argument(
        "--metrics-out",
        default="results/rolling_or_metrics.json",
        help="Path to the generated rolling OR metrics JSON.",
    )
    parser.add_argument(
        "--time-limit-sec",
        type=float,
        default=DEFAULT_ROLLING_TIME_LIMIT_SEC,
        help="Maximum solver runtime per rolling re-optimization call in seconds.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of CP-SAT search workers per rolling solve.",
    )
    return parser


def _format_solver_summary(metrics: dict[str, Any]) -> tuple[str, str]:
    """Build stable terminal summary lines for rolling solver diagnostics."""

    solve_call_count = int(metrics.get("solve_call_count", 0))
    avg_solve_time_sec = float(metrics.get("avg_solve_time_sec", 0.0))
    raw_status_counts = metrics.get("solver_status_counts", {})
    status_counts = raw_status_counts if isinstance(raw_status_counts, dict) else {}

    ordered_statuses = [
        f"{status}={int(status_counts.get(status, 0))}"
        for status in KNOWN_SOLVER_STATUSES
    ]
    extra_statuses = sorted(
        status for status in status_counts if status not in KNOWN_SOLVER_STATUSES
    )
    ordered_statuses.extend(
        f"{status}={int(status_counts[status])}" for status in extra_statuses
    )

    summary_line = (
        f"Solver summary: {solve_call_count} solves, "
        f"avg {avg_solve_time_sec:.4f}s per solve"
    )
    status_line = "Statuses: " + ", ".join(ordered_statuses)
    return summary_line, status_line


def main() -> None:
    """Replay the full horizon with repeated one-run OR solves and persist outputs."""

    args = build_parser().parse_args()
    result = run_rolling_or_model_from_csv(
        args.tickets,
        args.agents,
        time_limit_sec=args.time_limit_sec,
        num_workers=args.num_workers,
    )
    write_rolling_or_outputs(result, args.schedule_out, args.metrics_out)
    print("Wrote rolling OR outputs to " f"{args.schedule_out} and {args.metrics_out}")
    summary_line, status_line = _format_solver_summary(result.metrics)
    print(summary_line)
    print(status_line)


if __name__ == "__main__":
    main()
