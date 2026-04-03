"""Tests for the rolling-horizon OR replay controller."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.evaluation import SCHEDULE_FIELDNAMES
from src.or_rolling import (
    run_rolling_or_model,
    run_rolling_or_model_from_csv,
    write_rolling_or_outputs,
)
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


class TestRollingOrModel(unittest.TestCase):
    """Validate the rolling replay wrapper built on top of the one-run OR model."""

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

    def test_current_slot_only_commitment_can_change_future_plan(self) -> None:
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
                        "first_response_due_ts": "2026-03-02 08:15:00",
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
                        "first_response_due_ts": "2026-03-02 08:30:00",
                        "resolution_due_ts": "2026-03-02 09:00:00",
                    },
                ],
            )
            self.write_agents(agents_path, self.default_agents())

            result = run_rolling_or_model_from_csv(
                tickets_path, agents_path, time_limit_sec=2, num_workers=1
            )
            schedule_by_id = {entry.ticket_id: entry for entry in result.schedule}

            self.assertEqual(schedule_by_id["TKT-A"].status, "scheduled")
            self.assertEqual(schedule_by_id["TKT-A"].start_ts, datetime(2026, 3, 2, 8, 0))
            self.assertEqual(schedule_by_id["TKT-C"].status, "scheduled")
            self.assertEqual(
                schedule_by_id["TKT-C"].start_ts, datetime(2026, 3, 2, 8, 15)
            )
            self.assertEqual(schedule_by_id["TKT-B"].status, "scheduled")
            self.assertGreaterEqual(
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

            result = run_rolling_or_model_from_csv(
                tickets_path, agents_path, time_limit_sec=2, num_workers=1
            )
            schedule_by_id = {entry.ticket_id: entry for entry in result.schedule}

            self.assertEqual(schedule_by_id["TKT-D1"].status, "scheduled")
            self.assertEqual(
                schedule_by_id["TKT-D1"].start_ts, datetime(2026, 3, 3, 8, 0)
            )
            self.assertEqual(schedule_by_id["TKT-D2"].status, "scheduled")
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

            result = run_rolling_or_model_from_csv(
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

            result = run_rolling_or_model_from_csv(
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

            result = run_rolling_or_model_from_csv(
                tickets_path, agents_path, time_limit_sec=2, num_workers=1
            )
            write_rolling_or_outputs(result, schedule_path, metrics_path)

            metrics = result.metrics
            self.assertIn("replay_business_days", metrics)
            self.assertIn("slot_minutes", metrics)
            self.assertIn("scheduled", metrics)
            self.assertIn("backlog", metrics)
            self.assertIn("agent_utilization", metrics)
            self.assertIn("solve_call_count", metrics)
            self.assertIn("avg_solve_time_sec", metrics)
            self.assertIn("solver_status_counts", metrics)
            self.assertEqual(
                metrics["scheduled_tickets"], metrics["scheduled"]["ticket_count"]
            )
            self.assertEqual(
                metrics["tickets_in_backlog"], metrics["backlog"]["ticket_count"]
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

            direct_result = run_rolling_or_model(
                tickets, agents, time_limit_sec=2, num_workers=1
            )
            csv_result = run_rolling_or_model_from_csv(
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


if __name__ == "__main__":
    unittest.main()
