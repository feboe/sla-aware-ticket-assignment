"""Tests for the minimal one-run CP-SAT ticket-assignment model."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.greedy_baseline import SCHEDULE_FIELDNAMES
from src.or_model import (
    extract_or_schedule,
    prepare_or_instance,
    run_or_model_from_csv,
    solve_cp_sat_instance,
    write_or_outputs,
)

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


class TestOrModel(unittest.TestCase):
    """Validate preprocessing, exact decisions, and output contracts for the OR model."""

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

    def default_agents(self) -> list[dict[str, str]]:
        return [
            {
                "agent_id": "AG-01",
                "agent_role": "generalist",
                "languages": "EN",
                "queue_permissions": "Product Support",
                "priority_scope": "P1|P2|P3|P4",
                "skill_tags": "product_support",
                "shift_start": "08:00",
                "shift_end": "16:00",
                "capacity_min_per_day": "480",
                "scarce_resource": "0",
            }
        ]

    def test_prepare_instance_rounds_filters_and_limits_starts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tickets_path = Path(tmpdir) / "tickets.csv"
            agents_path = Path(tmpdir) / "agents.csv"
            self.write_tickets(
                tickets_path,
                [
                    {
                        "ticket_id": "TKT-01",
                        "arrival_ts": "2026-03-02 08:01:00",
                        "queue": "Product Support",
                        "priority": "P2",
                        "language": "EN",
                        "estimated_effort_min": "30",
                        "first_response_due_ts": "2026-03-02 09:00:00",
                        "resolution_due_ts": "2026-03-02 12:00:00",
                    },
                    {
                        "ticket_id": "TKT-02",
                        "arrival_ts": "2026-03-02 15:31:00",
                        "queue": "Product Support",
                        "priority": "P3",
                        "language": "EN",
                        "estimated_effort_min": "30",
                        "first_response_due_ts": "2026-03-02 16:30:00",
                        "resolution_due_ts": "2026-03-03 10:00:00",
                    },
                    {
                        "ticket_id": "TKT-03",
                        "arrival_ts": "2026-03-02 10:00:00",
                        "queue": "Product Support",
                        "priority": "P4",
                        "language": "EN",
                        "estimated_effort_min": "15",
                        "first_response_due_ts": "2026-03-02 11:00:00",
                        "resolution_due_ts": "2026-03-02 13:00:00",
                    },
                ],
            )
            self.write_agents(agents_path, self.default_agents())

            instance = prepare_or_instance(
                tickets_path, agents_path, "2026-03-02 08:07:00"
            )

            self.assertEqual(
                instance.requested_decision_ts, datetime(2026, 3, 2, 8, 7, 0)
            )
            self.assertEqual(instance.horizon_start_ts, datetime(2026, 3, 2, 8, 15, 0))
            self.assertEqual(instance.horizon_end_ts, datetime(2026, 3, 2, 16, 0, 0))
            self.assertEqual(
                [ticket.ticket_id for ticket in instance.tickets], ["TKT-01"]
            )
            self.assertEqual(
                instance.allowed_start_indices[("TKT-01", "AG-01")][:3], (0, 1, 2)
            )

            late_instance = prepare_or_instance(
                tickets_path, agents_path, "2026-03-02 15:45:00"
            )
            self.assertEqual(
                [ticket.ticket_id for ticket in late_instance.tickets],
                ["TKT-01", "TKT-03", "TKT-02"],
            )
            self.assertEqual(late_instance.allowed_start_indices[("TKT-02", "AG-01")], ())

    def test_single_ticket_is_scheduled(self) -> None:
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
                        "estimated_effort_min": "30",
                        "first_response_due_ts": "2026-03-02 08:30:00",
                        "resolution_due_ts": "2026-03-02 10:00:00",
                    }
                ],
            )
            self.write_agents(agents_path, self.default_agents())

            result = run_or_model_from_csv(
                tickets_path, agents_path, "2026-03-02 08:00:00"
            )

            self.assertEqual(len(result.schedule), 1)
            self.assertEqual(result.schedule[0].status, "scheduled")
            self.assertEqual(result.schedule[0].agent_id, "AG-01")
            self.assertEqual(result.schedule[0].start_ts, datetime(2026, 3, 2, 8, 0, 0))
            self.assertEqual(
                result.schedule[0].completion_ts, datetime(2026, 3, 2, 8, 30, 0)
            )
            self.assertEqual(result.metrics["scheduled_tickets"], 1)
            self.assertEqual(result.metrics["tickets_in_backlog"], 0)

    def test_infeasible_ticket_goes_to_backlog(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tickets_path = Path(tmpdir) / "tickets.csv"
            agents_path = Path(tmpdir) / "agents.csv"
            self.write_tickets(
                tickets_path,
                [
                    {
                        "ticket_id": "TKT-01",
                        "arrival_ts": "2026-03-02 08:00:00",
                        "queue": "Integrations/API",
                        "priority": "P2",
                        "language": "DE",
                        "estimated_effort_min": "30",
                        "first_response_due_ts": "2026-03-02 08:30:00",
                        "resolution_due_ts": "2026-03-02 10:00:00",
                    }
                ],
            )
            self.write_agents(agents_path, self.default_agents())

            result = run_or_model_from_csv(
                tickets_path, agents_path, "2026-03-02 08:00:00"
            )

            self.assertEqual(result.schedule[0].status, "backlog_current_run")
            self.assertEqual(result.schedule[0].agent_id, "")
            self.assertEqual(result.metrics["scheduled_tickets"], 0)
            self.assertEqual(result.metrics["tickets_in_backlog"], 1)

    def test_higher_priority_ticket_wins_when_only_one_fits(self) -> None:
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
                        "estimated_effort_min": "480",
                        "first_response_due_ts": "2026-03-02 08:30:00",
                        "resolution_due_ts": "2026-03-02 16:00:00",
                    },
                    {
                        "ticket_id": "TKT-P4",
                        "arrival_ts": "2026-03-02 08:00:00",
                        "queue": "Product Support",
                        "priority": "P4",
                        "language": "EN",
                        "estimated_effort_min": "480",
                        "first_response_due_ts": "2026-03-02 08:30:00",
                        "resolution_due_ts": "2026-03-02 16:00:00",
                    },
                ],
            )
            self.write_agents(agents_path, self.default_agents())

            result = run_or_model_from_csv(
                tickets_path, agents_path, "2026-03-02 08:00:00"
            )
            schedule_by_id = {entry.ticket_id: entry for entry in result.schedule}

            self.assertEqual(schedule_by_id["TKT-P1"].status, "scheduled")
            self.assertEqual(schedule_by_id["TKT-P4"].status, "backlog_current_run")

    def test_extracted_schedule_has_no_agent_overlap(self) -> None:
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
                        "estimated_effort_min": "240",
                        "first_response_due_ts": "2026-03-02 08:15:00",
                        "resolution_due_ts": "2026-03-02 12:00:00",
                    },
                    {
                        "ticket_id": "TKT-02",
                        "arrival_ts": "2026-03-02 08:00:00",
                        "queue": "Product Support",
                        "priority": "P2",
                        "language": "EN",
                        "estimated_effort_min": "240",
                        "first_response_due_ts": "2026-03-02 09:00:00",
                        "resolution_due_ts": "2026-03-02 16:00:00",
                    },
                ],
            )
            self.write_agents(agents_path, self.default_agents())

            instance = prepare_or_instance(
                tickets_path, agents_path, "2026-03-02 08:00:00"
            )
            artifacts = solve_cp_sat_instance(instance)
            schedule = extract_or_schedule(artifacts)

            self.assertEqual(
                [entry.status for entry in schedule], ["scheduled", "scheduled"]
            )
            self.assertLessEqual(schedule[0].completion_ts, schedule[1].start_ts)

    def test_metrics_schema_and_writer_match_contract(self) -> None:
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
                        "estimated_effort_min": "480",
                        "first_response_due_ts": "2026-03-02 08:30:00",
                        "resolution_due_ts": "2026-03-02 16:00:00",
                    },
                    {
                        "ticket_id": "TKT-02",
                        "arrival_ts": "2026-03-02 08:00:00",
                        "queue": "Product Support",
                        "priority": "P4",
                        "language": "EN",
                        "estimated_effort_min": "480",
                        "first_response_due_ts": "2026-03-02 08:30:00",
                        "resolution_due_ts": "2026-03-02 16:00:00",
                    },
                ],
            )
            self.write_agents(agents_path, self.default_agents())

            result = run_or_model_from_csv(
                tickets_path, agents_path, "2026-03-02 08:00:00"
            )
            write_or_outputs(result, schedule_path, metrics_path)

            metrics = result.metrics
            expected_keys = {
                "slot_minutes",
                "decision_ts",
                "horizon_start_ts",
                "horizon_end_ts",
                "total_tickets",
                "scheduled_tickets",
                "tickets_in_backlog",
                "total_first_response_tardiness_min",
                "total_resolution_tardiness_min",
                "solver_status",
                "objective_value",
                "solve_time_sec",
                "scheduled",
                "backlog",
                "agent_utilization",
            }
            self.assertEqual(set(metrics.keys()), expected_keys)
            self.assertEqual(metrics["decision_ts"], "2026-03-02 08:00:00")
            self.assertEqual(metrics["horizon_start_ts"], "2026-03-02 08:00:00")
            self.assertEqual(metrics["horizon_end_ts"], "2026-03-02 16:00:00")
            self.assertEqual(
                metrics["scheduled_tickets"], metrics["scheduled"]["ticket_count"]
            )
            self.assertEqual(
                metrics["tickets_in_backlog"], metrics["backlog"]["ticket_count"]
            )
            self.assertEqual(
                metrics["total_first_response_tardiness_min"],
                round(
                    metrics["scheduled"]["first_response_tardiness_min"]
                    + metrics["backlog"]["first_response_tardiness_min"],
                    2,
                ),
            )
            self.assertEqual(
                metrics["total_resolution_tardiness_min"],
                round(
                    metrics["scheduled"]["resolution_tardiness_min"]
                    + metrics["backlog"]["resolution_tardiness_min"],
                    2,
                ),
            )
            self.assertEqual(
                set(metrics["scheduled"].keys()), set(metrics["backlog"].keys())
            )
            self.assertEqual(
                set(metrics["scheduled"]["by_priority"].keys()),
                set(metrics["backlog"]["by_priority"].keys()),
            )
            self.assertEqual(
                set(metrics["agent_utilization"]["AG-01"].keys()),
                {"workload_minutes", "capacity_minutes", "utilization"},
            )
            self.assertEqual(
                metrics["agent_utilization"]["AG-01"]["capacity_minutes"], 480
            )

            with schedule_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, SCHEDULE_FIELDNAMES)
                self.assertEqual(len(list(reader)), 2)

            with metrics_path.open(encoding="utf-8") as handle:
                written_metrics = json.load(handle)
                self.assertEqual(written_metrics["total_tickets"], 2)
                self.assertIn("scheduled", written_metrics)
                self.assertIn("backlog", written_metrics)
                self.assertIn("agent_utilization", written_metrics)


if __name__ == "__main__":
    unittest.main()
