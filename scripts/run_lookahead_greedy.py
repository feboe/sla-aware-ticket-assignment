"""CLI entrypoint for running the look-ahead greedy ticket benchmark."""

from __future__ import annotations

import argparse

from src.lookahead_greedy import (
    run_lookahead_greedy_from_csv,
    write_lookahead_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the look-ahead replay script."""

    parser = argparse.ArgumentParser(
        description="Run the look-ahead greedy ticket-assignment benchmark."
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
        default="results/lookahead_greedy_schedule.csv",
        help="Path to the generated schedule CSV.",
    )
    parser.add_argument(
        "--metrics-out",
        default="results/lookahead_greedy_metrics.json",
        help="Path to the generated metrics JSON.",
    )
    return parser


def main() -> None:
    """Execute the look-ahead replay and persist its schedule and metrics."""

    args = build_parser().parse_args()
    result = run_lookahead_greedy_from_csv(args.tickets, args.agents)
    write_lookahead_outputs(result, args.schedule_out, args.metrics_out)
    print(
        "Wrote look-ahead greedy outputs to "
        f"{args.schedule_out} and {args.metrics_out}"
    )


if __name__ == "__main__":
    main()
