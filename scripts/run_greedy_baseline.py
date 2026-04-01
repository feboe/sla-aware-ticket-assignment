"""CLI entrypoint for running the greedy ticket-assignment baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Allow running the script directly from the repository root without installing
# the project as a package first.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.greedy_baseline import run_greedy_baseline_from_csv, write_baseline_outputs


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the baseline replay script."""

    parser = argparse.ArgumentParser(
        description="Run the greedy ticket-assignment baseline over the synthetic dataset."
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
        default="results/greedy_baseline_schedule.csv",
        help="Path to the generated schedule CSV.",
    )
    parser.add_argument(
        "--metrics-out",
        default="results/greedy_baseline_metrics.json",
        help="Path to the generated metrics JSON.",
    )
    return parser


def main() -> None:
    """Execute the baseline replay and persist its schedule and metrics."""

    args = build_parser().parse_args()
    result = run_greedy_baseline_from_csv(args.tickets, args.agents)
    write_baseline_outputs(result, args.schedule_out, args.metrics_out)
    print(
        "Wrote greedy baseline outputs to " f"{args.schedule_out} and {args.metrics_out}"
    )


if __name__ == "__main__":
    main()
