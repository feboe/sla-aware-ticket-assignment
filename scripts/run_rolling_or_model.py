"""CLI entrypoint for running the rolling-horizon CP-SAT ticket-assignment benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.or_rolling import run_rolling_or_model_from_csv, write_rolling_or_outputs


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
        default=0.2,
        help="Maximum solver runtime per rolling re-optimization call in seconds.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of CP-SAT search workers per rolling solve.",
    )
    return parser


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


if __name__ == "__main__":
    main()
