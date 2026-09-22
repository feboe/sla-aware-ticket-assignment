"""Tests for the online greedy baseline and its output contracts."""

import csv
import json
import tempfile
import unittest
from datetime import datetime, time
from pathlib import Path

import src.greedy_baseline as greedy_baseline
from src.evaluation import SCHEDULE_FIELDNAMES, ScheduleEntry
from src.greedy_baseline import (
    run_greedy_baseline,
    run_greedy_baseline_from_csv,
    write_baseline_outputs,
)
from src.preprocessing import (
    AgentRecord,
    TicketRecord,
    ceil_to_slot,
    load_agents,
    load_tickets,
    round_effort_to_slots,
)

TICKETS_PATH = Path("data/tickets.csv")
AGENTS_PATH = Path("data/agents.csv")


class TestGreedyBaseline(unittest.TestCase):
    """Check baseline determinism, policy behavior, and I/O contracts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tickets = load_tickets(TICKETS_PATH)
        cls.agents = load_agents(AGENTS_PATH)
        cls.result = run_greedy_baseline(cls.tickets, cls.agents)
        cls.schedule = cls.result.schedule
        cls.metrics = cls.result.metrics

    def test_run_is_deterministic(self) -> None:
        second_run = run_greedy_baseline(self.tickets, self.agents)
        self.assertEqual(
            [entry.to_row() for entry in self.schedule],
            [entry.to_row() for entry in second_run.schedule],
        )
        self.assertEqual(self.metrics, second_run.metrics)

    def test_explicit_public_api_and_schedule_row_contract(self) -> None:
        self.assertEqual(
            greedy_baseline.__all__,
            [
                "BaselineResult",
                "ticket_sort_key",
                "run_greedy_baseline",
                "compute_metrics",
                "write_baseline_outputs",
                "run_greedy_baseline_from_csv",
            ],
        )
        entry = ScheduleEntry(
            ticket_id="TKT-01",
            status="scheduled",
            agent_id="AG-01",
            start_ts=datetime(2026, 3, 2, 8, 0, 0),
            completion_ts=datetime(2026, 3, 2, 8, 15, 0),
            arrival_ts=datetime(2026, 3, 2, 7, 55, 0),
            queue="Product Support",
            priority="P2",
            language="EN",
            duration_slots=round_effort_to_slots(15),
            first_response_due_ts=datetime(2026, 3, 2, 8, 5, 0),
            resolution_due_ts=datetime(2026, 3, 2, 9, 0, 0),
            first_response_tardiness_min=0.0,
            resolution_tardiness_min=0.0,
        )
        self.assertEqual(entry.to_row()["start_ts"], "2026-03-02 08:00:00")

    def test_writer_creates_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            schedule_path = Path(tmpdir) / "schedule.csv"
            metrics_path = Path(tmpdir) / "metrics.json"
            write_baseline_outputs(self.result, schedule_path, metrics_path)

            with schedule_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, SCHEDULE_FIELDNAMES)
                rows = list(reader)
                self.assertEqual(len(rows), len(self.schedule))

            with metrics_path.open(encoding="utf-8") as handle:
                metrics = json.load(handle)
                self.assertEqual(metrics["total_tickets"], len(self.schedule))
                self.assertIn("scheduled", metrics)
                self.assertIn("backlog", metrics)
                self.assertIn("agent_utilization", metrics)
                self.assertIn("overall_agent_utilization", metrics)

    def test_backlog_carries_across_days_when_ticket_cannot_fit(self) -> None:
        agents = [
            AgentRecord(
                agent_id="AG-01",
                queue_permissions=frozenset({"Product Support"}),
                languages=frozenset({"EN"}),
                priority_scope=frozenset({"P3", "P4"}),
                shift_start=time(8, 0),
                shift_end=time(16, 0),
                capacity_min_per_day=480,
                scarce_resource=False,
            )
        ]
        ticket_a_arrival = datetime(2026, 3, 2, 15, 50, 0)
        ticket_b_arrival = datetime(2026, 3, 3, 8, 0, 0)
        tickets = [
            TicketRecord(
                ticket_id="TKT-A",
                arrival_ts=ticket_a_arrival,
                release_ts=ceil_to_slot(ticket_a_arrival),
                queue="Product Support",
                priority="P3",
                language="EN",
                estimated_effort_min=30,
                duration_slots=round_effort_to_slots(30),
                first_response_due_ts=datetime(2026, 3, 3, 9, 0, 0),
                resolution_due_ts=datetime(2026, 3, 3, 12, 0, 0),
            ),
            TicketRecord(
                ticket_id="TKT-B",
                arrival_ts=ticket_b_arrival,
                release_ts=ceil_to_slot(ticket_b_arrival),
                queue="Product Support",
                priority="P4",
                language="EN",
                estimated_effort_min=15,
                duration_slots=round_effort_to_slots(15),
                first_response_due_ts=datetime(2026, 3, 3, 10, 0, 0),
                resolution_due_ts=datetime(2026, 3, 3, 13, 0, 0),
            ),
        ]

        result = run_greedy_baseline(tickets, agents)
        schedule = {entry.ticket_id: entry for entry in result.schedule}

        self.assertEqual(schedule["TKT-A"].status, "scheduled")
        self.assertEqual(schedule["TKT-A"].start_ts, datetime(2026, 3, 3, 8, 0, 0))
        self.assertEqual(schedule["TKT-B"].status, "scheduled")
        self.assertGreater(schedule["TKT-B"].start_ts, schedule["TKT-A"].start_ts)

    def test_backlog_tardiness_is_reported_separately_at_horizon_end(self) -> None:
        agents = [
            AgentRecord(
                agent_id="AG-01",
                queue_permissions=frozenset({"Product Support"}),
                languages=frozenset({"EN"}),
                priority_scope=frozenset({"P1", "P2", "P3", "P4"}),
                shift_start=time(8, 0),
                shift_end=time(16, 0),
                capacity_min_per_day=480,
                scarce_resource=False,
            )
        ]
        scheduled_arrival = datetime(2026, 3, 2, 8, 0, 0)
        overdue_backlog_arrival = datetime(2026, 3, 2, 8, 15, 0)
        not_due_backlog_arrival = datetime(2026, 3, 2, 8, 30, 0)
        tickets = [
            TicketRecord(
                ticket_id="TKT-S",
                arrival_ts=scheduled_arrival,
                release_ts=ceil_to_slot(scheduled_arrival),
                queue="Product Support",
                priority="P1",
                language="EN",
                estimated_effort_min=480,
                duration_slots=round_effort_to_slots(480),
                first_response_due_ts=datetime(2026, 3, 2, 8, 15, 0),
                resolution_due_ts=datetime(2026, 3, 3, 8, 0, 0),
            ),
            TicketRecord(
                ticket_id="TKT-B1",
                arrival_ts=overdue_backlog_arrival,
                release_ts=ceil_to_slot(overdue_backlog_arrival),
                queue="Product Support",
                priority="P2",
                language="EN",
                estimated_effort_min=15,
                duration_slots=round_effort_to_slots(15),
                first_response_due_ts=datetime(2026, 3, 2, 8, 30, 0),
                resolution_due_ts=datetime(2026, 3, 2, 10, 0, 0),
            ),
            TicketRecord(
                ticket_id="TKT-B2",
                arrival_ts=not_due_backlog_arrival,
                release_ts=ceil_to_slot(not_due_backlog_arrival),
                queue="Product Support",
                priority="P4",
                language="EN",
                estimated_effort_min=15,
                duration_slots=round_effort_to_slots(15),
                first_response_due_ts=datetime(2026, 3, 3, 9, 0, 0),
                resolution_due_ts=datetime(2026, 3, 3, 12, 0, 0),
            ),
        ]

        result = run_greedy_baseline(tickets, agents)
        metrics = result.metrics
        scheduled_metrics = metrics["scheduled"]
        backlog_metrics = metrics["backlog"]
        final_horizon_end = datetime(2026, 3, 2, 16, 0, 0)

        self.assertEqual(
            metrics["horizon_start_ts"],
            datetime(2026, 3, 2, 8, 0, 0).strftime("%Y-%m-%d %H:%M:%S"),
        )
        self.assertEqual(
            metrics["horizon_end_ts"],
            final_horizon_end.strftime("%Y-%m-%d %H:%M:%S"),
        )
        self.assertEqual(metrics["scheduled_tickets"], 1)
        self.assertEqual(metrics["tickets_in_backlog"], 2)
        self.assertEqual(metrics["total_first_response_tardiness_min"], 450.0)
        self.assertEqual(metrics["total_resolution_tardiness_min"], 360.0)
        self.assertEqual(scheduled_metrics["ticket_count"], 1)
        self.assertEqual(scheduled_metrics["effort_min"], 480)
        self.assertEqual(scheduled_metrics["first_response_tardiness_min"], 0.0)
        self.assertEqual(scheduled_metrics["resolution_tardiness_min"], 0.0)
        self.assertEqual(scheduled_metrics["overdue_first_response_count"], 0)
        self.assertEqual(scheduled_metrics["overdue_resolution_count"], 0)
        self.assertEqual(
            scheduled_metrics["by_priority"]["P1"],
            {
                "ticket_count": 1,
                "effort_min": 480,
                "first_response_tardiness_min": 0.0,
                "resolution_tardiness_min": 0.0,
                "overdue_first_response_count": 0,
                "overdue_resolution_count": 0,
            },
        )
        self.assertEqual(backlog_metrics["ticket_count"], 2)
        self.assertEqual(backlog_metrics["effort_min"], 30)
        self.assertEqual(backlog_metrics["first_response_tardiness_min"], 450.0)
        self.assertEqual(backlog_metrics["resolution_tardiness_min"], 360.0)
        self.assertEqual(backlog_metrics["overdue_first_response_count"], 1)
        self.assertEqual(backlog_metrics["overdue_resolution_count"], 1)
        self.assertEqual(
            backlog_metrics["by_priority"]["P2"],
            {
                "ticket_count": 1,
                "effort_min": 15,
                "first_response_tardiness_min": 450.0,
                "resolution_tardiness_min": 360.0,
                "overdue_first_response_count": 1,
                "overdue_resolution_count": 1,
            },
        )
        self.assertEqual(
            backlog_metrics["by_priority"]["P4"],
            {
                "ticket_count": 1,
                "effort_min": 15,
                "first_response_tardiness_min": 0.0,
                "resolution_tardiness_min": 0.0,
                "overdue_first_response_count": 0,
                "overdue_resolution_count": 0,
            },
        )
        self.assertEqual(
            sum(
                values["ticket_count"]
                for values in backlog_metrics["by_priority"].values()
            ),
            backlog_metrics["ticket_count"],
        )
        self.assertEqual(
            sum(
                values["effort_min"] for values in backlog_metrics["by_priority"].values()
            ),
            backlog_metrics["effort_min"],
        )
        self.assertEqual(
            sum(
                values["first_response_tardiness_min"]
                for values in backlog_metrics["by_priority"].values()
            ),
            backlog_metrics["first_response_tardiness_min"],
        )
        self.assertEqual(
            sum(
                values["resolution_tardiness_min"]
                for values in backlog_metrics["by_priority"].values()
            ),
            backlog_metrics["resolution_tardiness_min"],
        )

    def test_from_csv_loader_path_matches_direct_run(self) -> None:
        from_csv = run_greedy_baseline_from_csv(TICKETS_PATH, AGENTS_PATH)
        self.assertEqual(
            [entry.to_row() for entry in from_csv.schedule],
            [entry.to_row() for entry in self.schedule],
        )


if __name__ == "__main__":
    unittest.main()
