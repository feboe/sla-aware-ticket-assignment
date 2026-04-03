"""CLI entrypoint for running the minimal one-run CP-SAT ticket-assignment model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.or_model import run_or_model_from_csv, write_or_outputs


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the one-run OR solve."""

    parser = argparse.ArgumentParser(
        description="Solve one decision timestamp with the CP-SAT ticket-assignment model."
    )
    parser.add_argument(
        "--decision-ts",
        required=True,
        help="Decision timestamp in YYYY-MM-DD HH:MM:SS format.",
    )
    parser.add_argument(
        "--tickets",
        default="data/ticket_assignment_tickets.csv",
        help="Path to the ticket-demand CSV.",
    )
    parser.add_argument(
        "--agents",
        default="data/agents.csv",
        help="Path to the agent-supply CSV.",
    )
    parser.add_argument(
        "--schedule-out",
        default="results/or_model_schedule.csv",
        help="Path to the generated OR schedule CSV.",
    )
    parser.add_argument(
        "--metrics-out",
        default="results/or_model_metrics.json",
        help="Path to the generated OR metrics JSON.",
    )
    parser.add_argument(
        "--time-limit-sec",
        type=int,
        default=30,
        help="Maximum solver runtime in seconds.",
    )
    return parser


def main() -> None:
    """Solve one explicit decision timestamp and persist its outputs."""

    args = build_parser().parse_args()
    result = run_or_model_from_csv(
        args.tickets,
        args.agents,
        args.decision_ts,
        time_limit_sec=args.time_limit_sec,
    )
    write_or_outputs(result, args.schedule_out, args.metrics_out)
    print("Wrote OR model outputs to " f"{args.schedule_out} and {args.metrics_out}")


if __name__ == "__main__":
    main()
