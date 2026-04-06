"""Tests for the current-slot OR scheduler and its CLI."""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from scripts import run_or_scheduler as scheduler_cli
from src.evaluation import SCHEDULE_FIELDNAMES
from src.greedy_baseline import run_greedy_baseline_from_csv
from src.lookahead_greedy import run_lookahead_greedy_from_csv
from src.or_scheduler import (
    OrSchedulerResult,
    run_or_scheduler,
    run_or_scheduler_from_csv,
    write_or_scheduler_outputs,
)
from src.or_preparation import prepare_or_scheduler_instance
from src.or_reporting import extract_or_scheduler_schedule
from src.or_solver import BACKLOG_WEIGHT, solve_or_scheduler_instance
from src.preprocessing import load_agents, load_tickets

TICKET_FIELDNAMES = [
    "ticket_id",
    "arrival_ts",
    "queue",
    "priority",
    "language",
    "estimated_effort_min",
    "first_response_due_ts",
    "resolution_due_ts",
]
AGENT_FIELDNAMES = [
    "agent_id",
    "agent_role",
    "languages",
    "queue_permissions",
    "priority_scope",
    "skill_tags",
    "shift_start",
    "shift_end",
    "capacity_min_per_day",
    "scarce_resource",
]


class TestOrScheduler(unittest.TestCase):
    """Validate the realistic current-slot OR scheduler workflow."""

    def write_tickets(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TICKET_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    def write_agents(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=AGENT_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    def default_agents(
        self,
        *,
        shift_start: str = "08:00",
        shift_end: str = "09:00",
        capacity_min_per_day: str = "60",
    ) -> list[dict[str, str]]:
        return [
            {
                "agent_id": "AG-01",
                "agent_role": "generalist",
                "languages": "EN",
                "queue_permissions": "Product Support",
                "priority_scope": "P1|P2|P3|P4",
                "skill_tags": "product_support",
                "shift_start": shift_start,
                "shift_end": shift_end,
                "capacity_min_per_day": capacity_min_per_day,
                "scarce_resource": "0",
            }
        ]

    def test_internal_model_has_no_future_start_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tickets_path = Path(tmpdir) / "tickets.csv"
            agents_path = Path(tmpdir) / "agents.csv"
            self.write_tickets(
                tickets_path,
                [
                    {
                        "ticket_id": "TKT-01",
                        "arrival_ts": "2026-03-02 08:00:00",
                        "queue": "Product Support",
                        "priority": "P2",
                        "language": "EN",
                        "estimated_effort_min": "15",
                        "first_response_due_ts": "2026-03-02 08:30:00",
                        "resolution_due_ts": "2026-03-02 09:00:00",
                    }
                ],
            )
            self.write_agents(agents_path, self.default_agents())

            instance = prepare_or_scheduler_instance(
                tickets_path, agents_path, "2026-03-02 08:00:00"
            )
            self.assertFalse(hasattr(instance, "slot_starts"))
            self.assertFalse(hasattr(instance, "allowed_start_indices"))
            self.assertEqual(instance.decision_ts, datetime(2026, 3, 2, 8, 0))
            self.assertEqual(instance.horizon_end_ts, datetime(2026, 3, 2, 9, 0))
            self.assertEqual(instance.horizon_slot_count, 4)
            self.assertEqual(instance.next_decision_ts, datetime(2026, 3, 2, 8, 15))

            artifacts = solve_or_scheduler_instance(
                instance,
                time_limit_sec=1,
                num_workers=1,
            )
            self.assertTrue(all(len(key) == 2 for key in artifacts.variables.x))
            self.assertEqual(
                set(artifacts.variables.first_response_tardiness),
                {"TKT-01"},
            )
            self.assertEqual(
                set(artifacts.variables.resolution_tardiness),
                {"TKT-01"},
            )
            schedule = extract_or_scheduler_schedule(artifacts)
            self.assertEqual(schedule[0].start_ts, datetime(2026, 3, 2, 8, 0))

    def test_backlog_uses_horizon_proxy_instead_of_next_slot_cost(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tickets_path = Path(tmpdir) / "tickets.csv"
            agents_path = Path(tmpdir) / "agents.csv"
            self.write_tickets(
                tickets_path,
                [
                    {
                        "ticket_id": "TKT-01",
                        "arrival_ts": "2026-03-02 08:00:00",
                        "queue": "Product Support",
                        "priority": "P2",
                        "language": "EN",
                        "estimated_effort_min": "15",
                        "first_response_due_ts": "2026-03-02 08:30:00",
                        "resolution_due_ts": "2026-03-02 09:00:00",
                    }
                ],
            )
            self.write_agents(
                agents_path,
                [
                    {
                        **self.default_agents()[0],
                        "languages": "DE",
                    }
                ],
            )

            instance = prepare_or_scheduler_instance(
                tickets_path, agents_path, "2026-03-02 08:00:00"
            )
            artifacts = solve_or_scheduler_instance(
                instance,
                time_limit_sec=1,
                num_workers=1,
            )

            self.assertEqual(artifacts.status_name, "OPTIMAL")
            self.assertEqual(artifacts.objective_value, 400.0 + BACKLOG_WEIGHT)

    def test_solver_prefers_non_scarce_agent_when_ticket_has_an_alternative(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tickets_path = Path(tmpdir) / "tickets.csv"
            agents_path = Path(tmpdir) / "agents.csv"
            self.write_tickets(
                tickets_path,
                [
                    {
                        "ticket_id": "TKT-01",
                        "arrival_ts": "2026-03-02 08:00:00",
                        "queue": "Product Support",
                        "priority": "P3",
                        "language": "EN",
                        "estimated_effort_min": "15",
                        "first_response_due_ts": "2026-03-02 10:00:00",
                        "resolution_due_ts": "2026-03-02 12:00:00",
                    }
                ],
            )
            self.write_agents(
                agents_path,
                [
                    {
                        **self.default_agents()[0],
                        "agent_id": "AG-NONSCARCE",
                        "scarce_resource": "0",
                    },
                    {
                        **self.default_agents()[0],
                        "agent_id": "AG-SCARCE",
                        "scarce_resource": "1",
                    },
                ],
            )

            instance = prepare_or_scheduler_instance(
                tickets_path, agents_path, "2026-03-02 08:00:00"
            )
            artifacts = solve_or_scheduler_instance(
                instance,
                time_limit_sec=1,
                num_workers=1,
            )
            schedule = extract_or_scheduler_schedule(artifacts)

            self.assertEqual(schedule[0].status, "scheduled")
            self.assertEqual(schedule[0].agent_id, "AG-NONSCARCE")

    def test_overdue_ticket_beats_less_urgent_ticket_when_only_one_start_fits(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tickets_path = Path(tmpdir) / "tickets.csv"
            agents_path = Path(tmpdir) / "agents.csv"
            self.write_tickets(
                tickets_path,
                [
                    {
                        "ticket_id": "TKT-P1",
                        "arrival_ts": "2026-03-02 08:00:00",
                        "queue": "Product Support",
                        "priority": "P1",
                        "language": "EN",
                        "estimated_effort_min": "15",
                        "first_response_due_ts": "2026-03-02 08:00:00",
                        "resolution_due_ts": "2026-03-02 08:30:00",
                    },
                    {
                        "ticket_id": "TKT-P4",
                        "arrival_ts": "2026-03-02 08:00:00",
                        "queue": "Product Support",
                        "priority": "P4",
                        "language": "EN",
                        "estimated_effort_min": "15",
                        "first_response_due_ts": "2026-03-02 10:00:00",
                        "resolution_due_ts": "2026-03-02 12:00:00",
                    },
                ],
            )
            self.write_agents(agents_path, self.default_agents())

            instance = prepare_or_scheduler_instance(
                tickets_path, agents_path, "2026-03-02 08:00:00"
            )
            artifacts = solve_or_scheduler_instance(
                instance,
                time_limit_sec=1,
                num_workers=1,
            )
            schedule_by_id = {
                entry.ticket_id: entry
                for entry in extract_or_scheduler_schedule(artifacts)
            }

            self.assertEqual(schedule_by_id["TKT-P1"].status, "scheduled")
            self.assertEqual(schedule_by_id["TKT-P4"].status, "backlog_current_run")

    def test_backlog_and_new_releases_are_reconsidered_each_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tickets_path = Path(tmpdir) / "tickets.csv"
            agents_path = Path(tmpdir) / "agents.csv"
            self.write_tickets(
                tickets_path,
                [
                    {
                        "ticket_id": "TKT-A",
                        "arrival_ts": "2026-03-02 08:00:00",
                        "queue": "Product Support",
                        "priority": "P1",
                        "language": "EN",
                        "estimated_effort_min": "15",
                        "first_response_due_ts": "2026-03-02 08:00:00",
                        "resolution_due_ts": "2026-03-02 09:00:00",
                    },
                    {
                        "ticket_id": "TKT-B",
                        "arrival_ts": "2026-03-02 08:00:00",
                        "queue": "Product Support",
                        "priority": "P4",
                        "language": "EN",
                        "estimated_effort_min": "15",
                        "first_response_due_ts": "2026-03-02 09:00:00",
                        "resolution_due_ts": "2026-03-02 10:00:00",
                    },
                    {
                        "ticket_id": "TKT-C",
                        "arrival_ts": "2026-03-02 08:15:00",
                        "queue": "Product Support",
                        "priority": "P1",
                        "language": "EN",
                        "estimated_effort_min": "15",
                        "first_response_due_ts": "2026-03-02 08:15:00",
                        "resolution_due_ts": "2026-03-02 09:00:00",
                    },
                ],
            )
            self.write_agents(agents_path, self.default_agents())

            result = run_or_scheduler_from_csv(
                tickets_path, agents_path, time_limit_sec=2, num_workers=1
            )
            schedule_by_id = {entry.ticket_id: entry for entry in result.schedule}

            self.assertEqual(schedule_by_id["TKT-A"].start_ts, datetime(2026, 3, 2, 8, 0))
            self.assertEqual(
                schedule_by_id["TKT-C"].start_ts, datetime(2026, 3, 2, 8, 15)
            )
            self.assertEqual(
                schedule_by_id["TKT-B"].start_ts, datetime(2026, 3, 2, 8, 30)
            )

    def test_backlog_carries_across_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tickets_path = Path(tmpdir) / "tickets.csv"
            agents_path = Path(tmpdir) / "agents.csv"
            self.write_tickets(
                tickets_path,
                [
                    {
                        "ticket_id": "TKT-D1",
                        "arrival_ts": "2026-03-02 08:45:00",
                        "queue": "Product Support",
                        "priority": "P1",
                        "language": "EN",
                        "estimated_effort_min": "30",
                        "first_response_due_ts": "2026-03-02 09:00:00",
                        "resolution_due_ts": "2026-03-03 10:00:00",
                    },
                    {
                        "ticket_id": "TKT-D2",
                        "arrival_ts": "2026-03-03 08:00:00",
                        "queue": "Product Support",
                        "priority": "P4",
                        "language": "EN",
                        "estimated_effort_min": "15",
                        "first_response_due_ts": "2026-03-03 09:00:00",
                        "resolution_due_ts": "2026-03-03 12:00:00",
                    },
                ],
            )
            self.write_agents(agents_path, self.default_agents())

            result = run_or_scheduler_from_csv(
                tickets_path, agents_path, time_limit_sec=2, num_workers=1
            )
            schedule_by_id = {entry.ticket_id: entry for entry in result.schedule}

            self.assertEqual(
                schedule_by_id["TKT-D1"].start_ts, datetime(2026, 3, 3, 8, 0)
            )
            self.assertGreaterEqual(
                schedule_by_id["TKT-D2"].start_ts, datetime(2026, 3, 3, 8, 30)
            )

    def test_previously_started_work_blocks_later_solves(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tickets_path = Path(tmpdir) / "tickets.csv"
            agents_path = Path(tmpdir) / "agents.csv"
            self.write_tickets(
                tickets_path,
                [
                    {
                        "ticket_id": "LONG",
                        "arrival_ts": "2026-03-02 08:00:00",
                        "queue": "Product Support",
                        "priority": "P1",
                        "language": "EN",
                        "estimated_effort_min": "30",
                        "first_response_due_ts": "2026-03-02 08:15:00",
                        "resolution_due_ts": "2026-03-02 09:00:00",
                    },
                    {
                        "ticket_id": "SHORT",
                        "arrival_ts": "2026-03-02 08:15:00",
                        "queue": "Product Support",
                        "priority": "P2",
                        "language": "EN",
                        "estimated_effort_min": "15",
                        "first_response_due_ts": "2026-03-02 08:45:00",
                        "resolution_due_ts": "2026-03-02 09:00:00",
                    },
                ],
            )
            self.write_agents(agents_path, self.default_agents())

            result = run_or_scheduler_from_csv(
                tickets_path, agents_path, time_limit_sec=2, num_workers=1
            )
            schedule_by_id = {entry.ticket_id: entry for entry in result.schedule}

            self.assertEqual(schedule_by_id["LONG"].start_ts, datetime(2026, 3, 2, 8, 0))
            self.assertEqual(
                schedule_by_id["SHORT"].start_ts, datetime(2026, 3, 2, 8, 30)
            )

    def test_same_day_completion_is_respected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tickets_path = Path(tmpdir) / "tickets.csv"
            agents_path = Path(tmpdir) / "agents.csv"
            self.write_tickets(
                tickets_path,
                [
                    {
                        "ticket_id": "TKT-LATE",
                        "arrival_ts": "2026-03-02 08:45:00",
                        "queue": "Product Support",
                        "priority": "P2",
                        "language": "EN",
                        "estimated_effort_min": "30",
                        "first_response_due_ts": "2026-03-02 09:00:00",
                        "resolution_due_ts": "2026-03-02 11:00:00",
                    }
                ],
            )
            self.write_agents(agents_path, self.default_agents())

            result = run_or_scheduler_from_csv(
                tickets_path, agents_path, time_limit_sec=2, num_workers=1
            )

            self.assertEqual(result.schedule[0].status, "backlog_end")
            self.assertIsNone(result.schedule[0].start_ts)

    def test_metrics_schema_matches_replay_plus_solver_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tickets_path = Path(tmpdir) / "tickets.csv"
            agents_path = Path(tmpdir) / "agents.csv"
            schedule_path = Path(tmpdir) / "schedule.csv"
            metrics_path = Path(tmpdir) / "metrics.json"
            self.write_tickets(
                tickets_path,
                [
                    {
                        "ticket_id": "TKT-01",
                        "arrival_ts": "2026-03-02 08:00:00",
                        "queue": "Product Support",
                        "priority": "P1",
                        "language": "EN",
                        "estimated_effort_min": "15",
                        "first_response_due_ts": "2026-03-02 08:15:00",
                        "resolution_due_ts": "2026-03-02 09:00:00",
                    },
                    {
                        "ticket_id": "TKT-02",
                        "arrival_ts": "2026-03-02 08:15:00",
                        "queue": "Product Support",
                        "priority": "P4",
                        "language": "EN",
                        "estimated_effort_min": "30",
                        "first_response_due_ts": "2026-03-02 09:00:00",
                        "resolution_due_ts": "2026-03-02 10:00:00",
                    },
                ],
            )
            self.write_agents(agents_path, self.default_agents())

            result = run_or_scheduler_from_csv(
                tickets_path, agents_path, time_limit_sec=2, num_workers=1
            )
            write_or_scheduler_outputs(result, schedule_path, metrics_path)

            metrics = result.metrics
            self.assertIn("replay_business_days", metrics)
            self.assertIn("slot_minutes", metrics)
            self.assertIn("scheduled", metrics)
            self.assertIn("backlog", metrics)
            self.assertIn("agent_utilization", metrics)
            self.assertIn("overall_agent_utilization", metrics)
            self.assertIn("solve_call_count", metrics)
            self.assertIn("avg_solve_time_sec", metrics)
            self.assertIn("solver_status_counts", metrics)
            self.assertEqual(
                metrics["scheduled_tickets"], metrics["scheduled"]["ticket_count"]
            )
            self.assertEqual(
                metrics["tickets_in_backlog"], metrics["backlog"]["ticket_count"]
            )
            self.assertEqual(
                metrics["overall_agent_utilization"]["workload_minutes"],
                sum(
                    values["workload_minutes"]
                    for values in metrics["agent_utilization"].values()
                ),
            )
            self.assertEqual(
                metrics["overall_agent_utilization"]["capacity_minutes"],
                sum(
                    values["capacity_minutes"]
                    for values in metrics["agent_utilization"].values()
                ),
            )
            self.assertEqual(
                metrics["overall_agent_utilization"]["workload_hours"],
                round(
                    metrics["overall_agent_utilization"]["workload_minutes"] / 60.0, 2
                ),
            )
            self.assertEqual(
                metrics["overall_agent_utilization"]["capacity_hours"],
                round(
                    metrics["overall_agent_utilization"]["capacity_minutes"] / 60.0, 2
                ),
            )
            solved_counts_by_agent = {agent_id: {"p1": 0, "p2": 0, "p3": 0, "p4": 0} for agent_id in metrics["agent_utilization"]}
            for entry in result.schedule:
                if entry.status != "scheduled":
                    continue
                solved_counts_by_agent[entry.agent_id][entry.priority.lower()] += 1

            for agent_id, agent_metrics in metrics["agent_utilization"].items():
                self.assertIn("p1_tickets_solved", agent_metrics)
                self.assertIn("p2_tickets_solved", agent_metrics)
                self.assertIn("p3_tickets_solved", agent_metrics)
                self.assertIn("p4_tickets_solved", agent_metrics)
                self.assertEqual(
                    agent_metrics["p1_tickets_solved"],
                    solved_counts_by_agent[agent_id]["p1"],
                )
                self.assertEqual(
                    agent_metrics["p2_tickets_solved"],
                    solved_counts_by_agent[agent_id]["p2"],
                )
                self.assertEqual(
                    agent_metrics["p3_tickets_solved"],
                    solved_counts_by_agent[agent_id]["p3"],
                )
                self.assertEqual(
                    agent_metrics["p4_tickets_solved"],
                    solved_counts_by_agent[agent_id]["p4"],
                )

            with schedule_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, SCHEDULE_FIELDNAMES)
                rows = list(reader)
                self.assertEqual(len(rows), 2)
                self.assertTrue(
                    all(row["status"] in {"scheduled", "backlog_end"} for row in rows)
                )

            with metrics_path.open(encoding="utf-8") as handle:
                written_metrics = json.load(handle)
                self.assertIn("solver_status_counts", written_metrics)

    def test_from_csv_matches_direct_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tickets_path = Path(tmpdir) / "tickets.csv"
            agents_path = Path(tmpdir) / "agents.csv"
            self.write_tickets(
                tickets_path,
                [
                    {
                        "ticket_id": "TKT-01",
                        "arrival_ts": "2026-03-02 08:00:00",
                        "queue": "Product Support",
                        "priority": "P2",
                        "language": "EN",
                        "estimated_effort_min": "15",
                        "first_response_due_ts": "2026-03-02 08:30:00",
                        "resolution_due_ts": "2026-03-02 09:00:00",
                    }
                ],
            )
            self.write_agents(agents_path, self.default_agents())
            tickets = load_tickets(tickets_path)
            agents = load_agents(agents_path)

            direct_result = run_or_scheduler(
                tickets, agents, time_limit_sec=2, num_workers=1
            )
            csv_result = run_or_scheduler_from_csv(
                tickets_path, agents_path, time_limit_sec=2, num_workers=1
            )

            self.assertEqual(
                [entry.to_row() for entry in direct_result.schedule],
                [entry.to_row() for entry in csv_result.schedule],
            )
            self.assertEqual(
                direct_result.metrics["scheduled_tickets"],
                csv_result.metrics["scheduled_tickets"],
            )
            self.assertEqual(
                direct_result.metrics["tickets_in_backlog"],
                csv_result.metrics["tickets_in_backlog"],
            )

    def test_shared_metric_blocks_match_greedy_and_lookahead_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tickets_path = Path(tmpdir) / "tickets.csv"
            agents_path = Path(tmpdir) / "agents.csv"
            self.write_tickets(
                tickets_path,
                [
                    {
                        "ticket_id": "TKT-01",
                        "arrival_ts": "2026-03-02 08:00:00",
                        "queue": "Product Support",
                        "priority": "P1",
                        "language": "EN",
                        "estimated_effort_min": "15",
                        "first_response_due_ts": "2026-03-02 08:15:00",
                        "resolution_due_ts": "2026-03-02 09:00:00",
                    },
                    {
                        "ticket_id": "TKT-02",
                        "arrival_ts": "2026-03-02 08:15:00",
                        "queue": "Product Support",
                        "priority": "P4",
                        "language": "EN",
                        "estimated_effort_min": "15",
                        "first_response_due_ts": "2026-03-02 09:00:00",
                        "resolution_due_ts": "2026-03-02 10:00:00",
                    },
                ],
            )
            self.write_agents(agents_path, self.default_agents())

            or_metrics = run_or_scheduler_from_csv(
                tickets_path, agents_path, time_limit_sec=2, num_workers=1
            ).metrics
            greedy_metrics = run_greedy_baseline_from_csv(tickets_path, agents_path).metrics
            lookahead_metrics = run_lookahead_greedy_from_csv(
                tickets_path, agents_path
            ).metrics

            shared_top_level_keys = {
                "replay_business_days",
                "slot_minutes",
                "horizon_start_ts",
                "horizon_end_ts",
                "total_tickets",
                "scheduled_tickets",
                "tickets_in_backlog",
                "total_first_response_tardiness_min",
                "total_resolution_tardiness_min",
                "scheduled",
                "backlog",
                "agent_utilization",
                "overall_agent_utilization",
            }
            self.assertTrue(shared_top_level_keys.issubset(or_metrics.keys()))
            self.assertEqual(set(greedy_metrics.keys()), shared_top_level_keys)
            self.assertEqual(set(lookahead_metrics.keys()), shared_top_level_keys)
            self.assertEqual(
                set(or_metrics["overall_agent_utilization"].keys()),
                set(greedy_metrics["overall_agent_utilization"].keys()),
            )
            self.assertEqual(
                set(or_metrics["overall_agent_utilization"].keys()),
                set(lookahead_metrics["overall_agent_utilization"].keys()),
            )

    def test_cli_prints_solver_summary(self) -> None:
        result = OrSchedulerResult(
            schedule=[],
            metrics={
                "solve_call_count": 155,
                "avg_solve_time_sec": 0.7,
                "solver_status_counts": {
                    "OPTIMAL": 58,
                    "FEASIBLE": 20,
                    "UNKNOWN": 77,
                },
            },
        )

        with (
            patch.object(scheduler_cli, "run_or_scheduler_from_csv", return_value=result),
            patch.object(scheduler_cli, "write_or_scheduler_outputs"),
            patch.object(
                sys,
                "argv",
                [
                    "run_or_scheduler.py",
                    "--schedule-out",
                    "tmp_schedule.csv",
                    "--metrics-out",
                    "tmp_metrics.json",
                ],
            ),
        ):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                scheduler_cli.main()

        output_lines = stdout.getvalue().strip().splitlines()
        self.assertEqual(
            output_lines[0],
            "Wrote OR scheduler outputs to tmp_schedule.csv and tmp_metrics.json",
        )
        self.assertEqual(
            output_lines[1], "Solver summary: 155 solves, avg 0.7000s per solve"
        )
        self.assertEqual(output_lines[2], "Statuses: OPTIMAL=58, FEASIBLE=20, UNKNOWN=77")

    def test_cli_prints_missing_known_statuses_as_zero_and_sorts_extras(self) -> None:
        result = OrSchedulerResult(
            schedule=[],
            metrics={
                "solve_call_count": 3,
                "avg_solve_time_sec": 0.125,
                "solver_status_counts": {
                    "OPTIMAL": 1,
                    "INFEASIBLE": 2,
                    "MODEL_INVALID": 4,
                },
            },
        )

        with (
            patch.object(scheduler_cli, "run_or_scheduler_from_csv", return_value=result),
            patch.object(scheduler_cli, "write_or_scheduler_outputs"),
            patch.object(sys, "argv", ["run_or_scheduler.py"]),
        ):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                scheduler_cli.main()

        output_lines = stdout.getvalue().strip().splitlines()
        self.assertEqual(
            output_lines[1], "Solver summary: 3 solves, avg 0.1250s per solve"
        )
        self.assertEqual(
            output_lines[2],
            "Statuses: OPTIMAL=1, FEASIBLE=0, UNKNOWN=0, INFEASIBLE=2, MODEL_INVALID=4",
        )


if __name__ == "__main__":
    unittest.main()
