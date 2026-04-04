"""Tests for the online greedy baseline and its output contracts."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, time, timedelta
from pathlib import Path

import src.evaluation as evaluation
import src.greedy_baseline as greedy_baseline
import src.preprocessing as preprocessing
from src.greedy_baseline import (
    AgentRecord,
    SCHEDULE_FIELDNAMES,
    ScheduleEntry,
    SLOT_MINUTES,
    TicketRecord,
    ceil_to_slot,
    load_agents,
    load_tickets,
    run_greedy_baseline,
    run_greedy_baseline_from_csv,
    tardiness_minutes,
    write_baseline_outputs,
)


TICKETS_PATH = Path("data/tickets.csv")
AGENTS_PATH = Path("data/agents.csv")


class TestGreedyBaseline(unittest.TestCase):
    """Check baseline determinism, feasibility, timing, and metric integrity."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tickets = load_tickets(TICKETS_PATH)
        cls.agents = load_agents(AGENTS_PATH)
        cls.result = run_greedy_baseline(cls.tickets, cls.agents)
        cls.schedule = cls.result.schedule
        cls.metrics = cls.result.metrics
        cls.ticket_by_id = {ticket.ticket_id: ticket for ticket in cls.tickets}
        cls.agent_by_id = {agent.agent_id: agent for agent in cls.agents}

    def test_run_is_deterministic(self) -> None:
        second_run = run_greedy_baseline(self.tickets, self.agents)
        self.assertEqual(
            [entry.to_row() for entry in self.schedule],
            [entry.to_row() for entry in second_run.schedule],
        )
        self.assertEqual(self.metrics, second_run.metrics)

    def test_shared_symbols_are_reexported_from_greedy_module(self) -> None:
        self.assertIs(greedy_baseline.TicketRecord, preprocessing.TicketRecord)
        self.assertIs(greedy_baseline.AgentRecord, preprocessing.AgentRecord)
        self.assertIs(greedy_baseline.ScheduleEntry, evaluation.ScheduleEntry)
        self.assertIs(greedy_baseline.ceil_to_slot, preprocessing.ceil_to_slot)
        self.assertIs(greedy_baseline.load_tickets, preprocessing.load_tickets)
        self.assertIs(greedy_baseline.load_agents, preprocessing.load_agents)
        self.assertIs(greedy_baseline.tardiness_minutes, evaluation.tardiness_minutes)
        self.assertEqual(greedy_baseline.SLOT_MINUTES, preprocessing.SLOT_MINUTES)
        self.assertEqual(
            greedy_baseline.SCHEDULE_FIELDNAMES, evaluation.SCHEDULE_FIELDNAMES
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
            duration_slots=1,
            first_response_due_ts=datetime(2026, 3, 2, 8, 5, 0),
            resolution_due_ts=datetime(2026, 3, 2, 9, 0, 0),
            first_response_tardiness_min=0.0,
            resolution_tardiness_min=0.0,
        )
        self.assertEqual(entry.to_row()["start_ts"], "2026-03-02 08:00:00")

    def test_schedule_covers_all_tickets(self) -> None:
        self.assertEqual(len(self.schedule), len(self.tickets))
        self.assertEqual(
            {entry.ticket_id for entry in self.schedule},
            {ticket.ticket_id for ticket in self.tickets},
        )
        self.assertTrue(
            {entry.status for entry in self.schedule}.issubset(
                {"scheduled", "backlog_end"}
            )
        )

    def test_assigned_tickets_respect_feasibility_and_timing(self) -> None:
        for entry in self.schedule:
            ticket = self.ticket_by_id[entry.ticket_id]
            if entry.status != "scheduled":
                self.assertEqual(entry.agent_id, "")
                self.assertIsNone(entry.start_ts)
                self.assertIsNone(entry.completion_ts)
                continue

            agent = self.agent_by_id[entry.agent_id]
            self.assertIn(ticket.queue, agent.queue_permissions)
            self.assertIn(ticket.language, agent.languages)
            self.assertIn(ticket.priority, agent.priority_scope)
            self.assertGreaterEqual(entry.start_ts, ticket.arrival_ts)
            self.assertGreaterEqual(entry.start_ts, ticket.release_ts)
            self.assertEqual(entry.start_ts.second, 0)
            self.assertEqual(entry.start_ts.minute % SLOT_MINUTES, 0)
            self.assertEqual(entry.start_ts.date(), entry.completion_ts.date())
            self.assertEqual(
                entry.completion_ts - entry.start_ts,
                timedelta(minutes=ticket.duration_slots * SLOT_MINUTES),
            )
            self.assertGreaterEqual(
                entry.start_ts,
                datetime.combine(entry.start_ts.date(), agent.shift_start),
            )
            self.assertLessEqual(
                entry.completion_ts,
                datetime.combine(entry.start_ts.date(), agent.shift_end),
            )
            self.assertEqual(
                entry.first_response_tardiness_min,
                tardiness_minutes(entry.start_ts, ticket.first_response_due_ts),
            )
            self.assertEqual(
                entry.resolution_tardiness_min,
                tardiness_minutes(entry.completion_ts, ticket.resolution_due_ts),
            )

    def test_no_double_booking_and_daily_capacity(self) -> None:
        occupied_slots: set[tuple[str, datetime]] = set()
        daily_minutes: dict[tuple[str, datetime.date], int] = {}

        for entry in self.schedule:
            if entry.status != "scheduled":
                continue

            agent = self.agent_by_id[entry.agent_id]
            current = entry.start_ts
            while current < entry.completion_ts:
                key = (entry.agent_id, current)
                self.assertNotIn(key, occupied_slots)
                occupied_slots.add(key)
                current += timedelta(minutes=SLOT_MINUTES)

            day_key = (entry.agent_id, entry.start_ts.date())
            daily_minutes[day_key] = daily_minutes.get(day_key, 0) + (
                entry.duration_slots * SLOT_MINUTES
            )
            self.assertLessEqual(daily_minutes[day_key], agent.capacity_min_per_day)

    def test_backlog_metrics_match_schedule(self) -> None:
        backlog_entries = [
            entry for entry in self.schedule if entry.status == "backlog_end"
        ]
        self.assertEqual(len(backlog_entries), self.metrics["tickets_in_backlog"])
        self.assertEqual(
            sum(entry.duration_slots * SLOT_MINUTES for entry in backlog_entries),
            self.metrics["backlog"]["effort_min"],
        )
        backlog_metrics = self.metrics["backlog"]
        self.assertEqual(len(backlog_entries), backlog_metrics["ticket_count"])
        self.assertEqual(
            sum(entry.duration_slots * SLOT_MINUTES for entry in backlog_entries),
            backlog_metrics["effort_min"],
        )

    def test_metric_overview_matches_nested_sections(self) -> None:
        expected_keys = {
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
        }
        self.assertEqual(set(self.metrics.keys()), expected_keys)
        self.assertEqual(self.metrics["horizon_start_ts"], "2026-03-02 08:00:00")
        self.assertEqual(
            self.metrics["scheduled_tickets"],
            self.metrics["scheduled"]["ticket_count"],
        )
        self.assertEqual(
            self.metrics["tickets_in_backlog"],
            self.metrics["backlog"]["ticket_count"],
        )
        self.assertEqual(
            self.metrics["total_first_response_tardiness_min"],
            round(
                self.metrics["scheduled"]["first_response_tardiness_min"]
                + self.metrics["backlog"]["first_response_tardiness_min"],
                2,
            ),
        )
        self.assertEqual(
            self.metrics["total_resolution_tardiness_min"],
            round(
                self.metrics["scheduled"]["resolution_tardiness_min"]
                + self.metrics["backlog"]["resolution_tardiness_min"],
                2,
            ),
        )
        self.assertNotIn("scheduled_first_response_tardiness_min", self.metrics)
        self.assertNotIn("scheduled_resolution_tardiness_min", self.metrics)
        self.assertNotIn("backlog_tardiness_at_horizon_end", self.metrics)
        self.assertNotIn("sla_violation_counts_by_priority", self.metrics)
        self.assertNotIn("workload_minutes_per_agent", self.metrics)
        self.assertNotIn("utilization_per_agent", self.metrics)

    def test_scheduled_and_backlog_sections_share_schema(self) -> None:
        scheduled = self.metrics["scheduled"]
        backlog = self.metrics["backlog"]
        self.assertEqual(set(scheduled.keys()), set(backlog.keys()))
        self.assertEqual(
            set(scheduled["by_priority"].keys()),
            set(backlog["by_priority"].keys()),
        )

        for priority in scheduled["by_priority"]:
            self.assertEqual(
                set(scheduled["by_priority"][priority].keys()),
                set(backlog["by_priority"][priority].keys()),
            )

        for section in (scheduled, backlog):
            self.assertEqual(
                sum(values["ticket_count"] for values in section["by_priority"].values()),
                section["ticket_count"],
            )
            self.assertEqual(
                sum(values["effort_min"] for values in section["by_priority"].values()),
                section["effort_min"],
            )
            self.assertEqual(
                sum(
                    values["first_response_tardiness_min"]
                    for values in section["by_priority"].values()
                ),
                section["first_response_tardiness_min"],
            )
            self.assertEqual(
                sum(
                    values["resolution_tardiness_min"]
                    for values in section["by_priority"].values()
                ),
                section["resolution_tardiness_min"],
            )

    def test_agent_utilization_section_matches_schedule(self) -> None:
        observed_agents = set(self.metrics["agent_utilization"].keys())
        self.assertEqual(observed_agents, set(self.agent_by_id.keys()))

        workload_by_agent = {agent_id: 0 for agent_id in self.agent_by_id}
        for entry in self.schedule:
            if entry.status != "scheduled":
                continue
            workload_by_agent[entry.agent_id] += entry.duration_slots * SLOT_MINUTES

        for agent_id, metrics in self.metrics["agent_utilization"].items():
            agent = self.agent_by_id[agent_id]
            capacity_minutes = (
                self.metrics["replay_business_days"] * agent.capacity_min_per_day
            )
            self.assertEqual(
                set(metrics.keys()),
                {"workload_minutes", "capacity_minutes", "utilization"},
            )
            self.assertEqual(metrics["workload_minutes"], workload_by_agent[agent_id])
            self.assertEqual(metrics["capacity_minutes"], capacity_minutes)
            self.assertEqual(
                metrics["utilization"],
                round(workload_by_agent[agent_id] / capacity_minutes, 4),
            )

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
                duration_slots=2,
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
                duration_slots=1,
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
                duration_slots=32,
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
                duration_slots=1,
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
                duration_slots=1,
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
