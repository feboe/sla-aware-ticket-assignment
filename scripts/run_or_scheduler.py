"""CLI entrypoint for running the OR scheduler benchmark."""

from __future__ import annotations

import argparse
from typing import Any

from src.or_scheduler import (
    DEFAULT_OR_SCHEDULER_TIME_LIMIT_SEC,
    run_or_scheduler_from_csv,
    write_or_scheduler_outputs,
)

KNOWN_SOLVER_STATUSES = ("OPTIMAL", "FEASIBLE", "UNKNOWN")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the OR scheduler replay."""

    parser = argparse.ArgumentParser(
        description="Replay the dataset with the current-slot OR scheduler."
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
        default="results/or_scheduler_schedule.csv",
        help="Path to the generated OR scheduler schedule CSV.",
    )
    parser.add_argument(
        "--metrics-out",
        default="results/or_scheduler_metrics.json",
        help="Path to the generated OR scheduler metrics JSON.",
    )
    parser.add_argument(
        "--time-limit-sec",
        type=float,
        default=DEFAULT_OR_SCHEDULER_TIME_LIMIT_SEC,
        help="Maximum solver runtime per scheduler call in seconds.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of CP-SAT search workers per scheduler solve.",
    )
    return parser


def _format_solver_summary(metrics: dict[str, Any]) -> tuple[str, str]:
    """Build stable terminal summary lines for scheduler solver diagnostics."""

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
    """Replay the full horizon with the OR scheduler and persist outputs."""

    args = build_parser().parse_args()
    result = run_or_scheduler_from_csv(
        args.tickets,
        args.agents,
        time_limit_sec=args.time_limit_sec,
        num_workers=args.num_workers,
    )
    write_or_scheduler_outputs(result, args.schedule_out, args.metrics_out)
    print("Wrote OR scheduler outputs to " f"{args.schedule_out} and {args.metrics_out}")
    summary_line, status_line = _format_solver_summary(result.metrics)
    print(summary_line)
    print(status_line)


if __name__ == "__main__":
    main()
