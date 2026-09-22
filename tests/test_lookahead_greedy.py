"""Tests for the same-day look-ahead greedy ticket-assignment benchmark."""

import csv
import json
import tempfile
import unittest
from datetime import datetime, time
from pathlib import Path

from src.evaluation import SCHEDULE_FIELDNAMES, write_schedule_outputs
from src.greedy_baseline import run_greedy_baseline
from src.lookahead_greedy import (
    run_lookahead_greedy,
    run_lookahead_greedy_from_csv,
)
from src.preprocessing import (
    AgentRecord,
    TicketRecord,
    load_agents,
    load_tickets,
    round_effort_to_slots,
)

TICKETS_PATH = Path("data/tickets.csv")
AGENTS_PATH = Path("data/agents.csv")


class TestLookaheadGreedy(unittest.TestCase):
    """Check determinism, I/O, and future-slot reservation behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tickets = load_tickets(TICKETS_PATH)
        cls.agents = load_agents(AGENTS_PATH)
        cls.result = run_lookahead_greedy(cls.tickets, cls.agents)
        cls.schedule = cls.result.schedule
        cls.metrics = cls.result.metrics

    def test_run_is_deterministic(self) -> None:
        second_run = run_lookahead_greedy(self.tickets, self.agents)
        self.assertEqual(
            [entry.to_row() for entry in self.schedule],
            [entry.to_row() for entry in second_run.schedule],
        )
        self.assertEqual(self.metrics, second_run.metrics)

    def test_writer_creates_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            schedule_path = Path(tmpdir) / "schedule.csv"
            metrics_path = Path(tmpdir) / "metrics.json"
            write_schedule_outputs(
                self.result.schedule,
                self.result.metrics,
                schedule_path,
                metrics_path,
            )

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

    def test_lookahead_can_schedule_future_same_day_slot(self) -> None:
        agents = [
            AgentRecord(
                agent_id="AG-01",
                queue_permissions=frozenset({"Product Support"}),
                languages=frozenset({"EN"}),
                priority_scope=frozenset({"P1", "P2", "P3", "P4"}),
                shift_start=time(8, 0),
                shift_end=time(8, 45),
                capacity_min_per_day=45,
                scarce_resource=False,
            )
        ]
        tickets = [
            TicketRecord(
                ticket_id="TKT-A",
                arrival_ts=datetime(2026, 3, 2, 8, 0, 0),
                release_ts=datetime(2026, 3, 2, 8, 0, 0),
                queue="Product Support",
                priority="P1",
                language="EN",
                estimated_effort_min=15,
                duration_slots=round_effort_to_slots(15),
                first_response_due_ts=datetime(2026, 3, 2, 8, 15, 0),
                resolution_due_ts=datetime(2026, 3, 2, 8, 45, 0),
            ),
            TicketRecord(
                ticket_id="TKT-B",
                arrival_ts=datetime(2026, 3, 2, 8, 0, 0),
                release_ts=datetime(2026, 3, 2, 8, 0, 0),
                queue="Product Support",
                priority="P4",
                language="EN",
                estimated_effort_min=15,
                duration_slots=round_effort_to_slots(15),
                first_response_due_ts=datetime(2026, 3, 2, 8, 30, 0),
                resolution_due_ts=datetime(2026, 3, 2, 9, 0, 0),
            ),
            TicketRecord(
                ticket_id="TKT-C",
                arrival_ts=datetime(2026, 3, 2, 8, 15, 0),
                release_ts=datetime(2026, 3, 2, 8, 15, 0),
                queue="Product Support",
                priority="P2",
                language="EN",
                estimated_effort_min=30,
                duration_slots=round_effort_to_slots(30),
                first_response_due_ts=datetime(2026, 3, 2, 8, 45, 0),
                resolution_due_ts=datetime(2026, 3, 2, 9, 30, 0),
            ),
        ]

        result = run_lookahead_greedy(tickets, agents)
        schedule = {entry.ticket_id: entry for entry in result.schedule}

        self.assertEqual(schedule["TKT-A"].start_ts, datetime(2026, 3, 2, 8, 0, 0))
        self.assertEqual(schedule["TKT-B"].status, "scheduled")
        self.assertEqual(schedule["TKT-B"].start_ts, datetime(2026, 3, 2, 8, 15, 0))
        self.assertEqual(schedule["TKT-C"].status, "backlog_end")

    def test_ticket_that_cannot_fit_later_today_stays_in_backlog(self) -> None:
        agents = [
            AgentRecord(
                agent_id="AG-01",
                queue_permissions=frozenset({"Product Support"}),
                languages=frozenset({"EN"}),
                priority_scope=frozenset({"P1", "P2", "P3", "P4"}),
                shift_start=time(8, 0),
                shift_end=time(8, 30),
                capacity_min_per_day=30,
                scarce_resource=False,
            )
        ]
        tickets = [
            TicketRecord(
                ticket_id="TKT-A",
                arrival_ts=datetime(2026, 3, 2, 8, 0, 0),
                release_ts=datetime(2026, 3, 2, 8, 0, 0),
                queue="Product Support",
                priority="P1",
                language="EN",
                estimated_effort_min=30,
                duration_slots=round_effort_to_slots(30),
                first_response_due_ts=datetime(2026, 3, 2, 8, 15, 0),
                resolution_due_ts=datetime(2026, 3, 2, 8, 45, 0),
            ),
            TicketRecord(
                ticket_id="TKT-B",
                arrival_ts=datetime(2026, 3, 2, 8, 0, 0),
                release_ts=datetime(2026, 3, 2, 8, 0, 0),
                queue="Product Support",
                priority="P3",
                language="EN",
                estimated_effort_min=15,
                duration_slots=round_effort_to_slots(15),
                first_response_due_ts=datetime(2026, 3, 2, 8, 30, 0),
                resolution_due_ts=datetime(2026, 3, 2, 9, 0, 0),
            ),
        ]

        result = run_lookahead_greedy(tickets, agents)
        schedule = {entry.ticket_id: entry for entry in result.schedule}
        self.assertEqual(schedule["TKT-A"].status, "scheduled")
        self.assertEqual(schedule["TKT-B"].status, "backlog_end")

    def test_two_tickets_can_be_reserved_back_to_back_on_same_agent(self) -> None:
        agents = [
            AgentRecord(
                agent_id="AG-01",
                queue_permissions=frozenset({"Product Support"}),
                languages=frozenset({"EN"}),
                priority_scope=frozenset({"P1", "P2", "P3", "P4"}),
                shift_start=time(8, 0),
                shift_end=time(9, 0),
                capacity_min_per_day=60,
                scarce_resource=False,
            )
        ]
        tickets = [
            TicketRecord(
                ticket_id="TKT-A",
                arrival_ts=datetime(2026, 3, 2, 8, 0, 0),
                release_ts=datetime(2026, 3, 2, 8, 0, 0),
                queue="Product Support",
                priority="P1",
                language="EN",
                estimated_effort_min=15,
                duration_slots=round_effort_to_slots(15),
                first_response_due_ts=datetime(2026, 3, 2, 8, 15, 0),
                resolution_due_ts=datetime(2026, 3, 2, 8, 45, 0),
            ),
            TicketRecord(
                ticket_id="TKT-B",
                arrival_ts=datetime(2026, 3, 2, 8, 0, 0),
                release_ts=datetime(2026, 3, 2, 8, 0, 0),
                queue="Product Support",
                priority="P3",
                language="EN",
                estimated_effort_min=15,
                duration_slots=round_effort_to_slots(15),
                first_response_due_ts=datetime(2026, 3, 2, 8, 45, 0),
                resolution_due_ts=datetime(2026, 3, 2, 9, 0, 0),
            ),
            TicketRecord(
                ticket_id="TKT-C",
                arrival_ts=datetime(2026, 3, 2, 8, 0, 0),
                release_ts=datetime(2026, 3, 2, 8, 0, 0),
                queue="Product Support",
                priority="P4",
                language="EN",
                estimated_effort_min=15,
                duration_slots=round_effort_to_slots(15),
                first_response_due_ts=datetime(2026, 3, 2, 9, 0, 0),
                resolution_due_ts=datetime(2026, 3, 2, 9, 15, 0),
            ),
        ]

        result = run_lookahead_greedy(tickets, agents)
        schedule = {entry.ticket_id: entry for entry in result.schedule}

        self.assertEqual(schedule["TKT-A"].start_ts, datetime(2026, 3, 2, 8, 0, 0))
        self.assertEqual(schedule["TKT-B"].start_ts, datetime(2026, 3, 2, 8, 15, 0))
        self.assertEqual(schedule["TKT-C"].start_ts, datetime(2026, 3, 2, 8, 30, 0))

    def test_future_reservations_block_later_overlap(self) -> None:
        agents = [
            AgentRecord(
                agent_id="AG-01",
                queue_permissions=frozenset({"Product Support"}),
                languages=frozenset({"EN"}),
                priority_scope=frozenset({"P1", "P2", "P3", "P4"}),
                shift_start=time(8, 0),
                shift_end=time(9, 0),
                capacity_min_per_day=60,
                scarce_resource=False,
            )
        ]
        tickets = [
            TicketRecord(
                ticket_id="TKT-A",
                arrival_ts=datetime(2026, 3, 2, 8, 0, 0),
                release_ts=datetime(2026, 3, 2, 8, 0, 0),
                queue="Product Support",
                priority="P1",
                language="EN",
                estimated_effort_min=15,
                duration_slots=round_effort_to_slots(15),
                first_response_due_ts=datetime(2026, 3, 2, 8, 15, 0),
                resolution_due_ts=datetime(2026, 3, 2, 8, 45, 0),
            ),
            TicketRecord(
                ticket_id="TKT-B",
                arrival_ts=datetime(2026, 3, 2, 8, 0, 0),
                release_ts=datetime(2026, 3, 2, 8, 0, 0),
                queue="Product Support",
                priority="P3",
                language="EN",
                estimated_effort_min=15,
                duration_slots=round_effort_to_slots(15),
                first_response_due_ts=datetime(2026, 3, 2, 8, 30, 0),
                resolution_due_ts=datetime(2026, 3, 2, 8, 45, 0),
            ),
            TicketRecord(
                ticket_id="TKT-C",
                arrival_ts=datetime(2026, 3, 2, 8, 15, 0),
                release_ts=datetime(2026, 3, 2, 8, 15, 0),
                queue="Product Support",
                priority="P2",
                language="EN",
                estimated_effort_min=15,
                duration_slots=round_effort_to_slots(15),
                first_response_due_ts=datetime(2026, 3, 2, 8, 45, 0),
                resolution_due_ts=datetime(2026, 3, 2, 9, 0, 0),
            ),
        ]

        result = run_lookahead_greedy(tickets, agents)
        schedule = {entry.ticket_id: entry for entry in result.schedule}

        self.assertEqual(schedule["TKT-B"].start_ts, datetime(2026, 3, 2, 8, 15, 0))
        self.assertEqual(schedule["TKT-C"].start_ts, datetime(2026, 3, 2, 8, 30, 0))

    def test_non_scarce_agent_is_preferred_on_same_earliest_slot(self) -> None:
        agents = [
            AgentRecord(
                agent_id="AG-02",
                queue_permissions=frozenset({"Product Support"}),
                languages=frozenset({"EN"}),
                priority_scope=frozenset({"P1", "P2", "P3", "P4"}),
                shift_start=time(8, 0),
                shift_end=time(16, 0),
                capacity_min_per_day=480,
                scarce_resource=True,
            ),
            AgentRecord(
                agent_id="AG-01",
                queue_permissions=frozenset({"Product Support"}),
                languages=frozenset({"EN"}),
                priority_scope=frozenset({"P1", "P2", "P3", "P4"}),
                shift_start=time(8, 0),
                shift_end=time(16, 0),
                capacity_min_per_day=480,
                scarce_resource=False,
            ),
        ]
        tickets = [
            TicketRecord(
                ticket_id="TKT-01",
                arrival_ts=datetime(2026, 3, 2, 8, 0, 0),
                release_ts=datetime(2026, 3, 2, 8, 0, 0),
                queue="Product Support",
                priority="P2",
                language="EN",
                estimated_effort_min=15,
                duration_slots=round_effort_to_slots(15),
                first_response_due_ts=datetime(2026, 3, 2, 8, 30, 0),
                resolution_due_ts=datetime(2026, 3, 2, 9, 0, 0),
            )
        ]

        result = run_lookahead_greedy(tickets, agents)
        self.assertEqual(result.schedule[0].agent_id, "AG-01")

    def test_lookahead_outperforms_myopic_on_future_placement_case(self) -> None:
        agents = [
            AgentRecord(
                agent_id="AG-01",
                queue_permissions=frozenset({"Product Support"}),
                languages=frozenset({"EN"}),
                priority_scope=frozenset({"P1", "P2", "P3", "P4"}),
                shift_start=time(8, 0),
                shift_end=time(9, 0),
                capacity_min_per_day=60,
                scarce_resource=False,
            )
        ]
        tickets = [
            TicketRecord(
                ticket_id="TKT-A",
                arrival_ts=datetime(2026, 3, 2, 8, 0, 0),
                release_ts=datetime(2026, 3, 2, 8, 0, 0),
                queue="Product Support",
                priority="P1",
                language="EN",
                estimated_effort_min=15,
                duration_slots=round_effort_to_slots(15),
                first_response_due_ts=datetime(2026, 3, 2, 8, 15, 0),
                resolution_due_ts=datetime(2026, 3, 2, 8, 45, 0),
            ),
            TicketRecord(
                ticket_id="TKT-B",
                arrival_ts=datetime(2026, 3, 2, 8, 0, 0),
                release_ts=datetime(2026, 3, 2, 8, 0, 0),
                queue="Product Support",
                priority="P4",
                language="EN",
                estimated_effort_min=15,
                duration_slots=round_effort_to_slots(15),
                first_response_due_ts=datetime(2026, 3, 2, 8, 30, 0),
                resolution_due_ts=datetime(2026, 3, 2, 9, 0, 0),
            ),
            TicketRecord(
                ticket_id="TKT-C",
                arrival_ts=datetime(2026, 3, 2, 8, 0, 0),
                release_ts=datetime(2026, 3, 2, 8, 0, 0),
                queue="Product Support",
                priority="P4",
                language="EN",
                estimated_effort_min=15,
                duration_slots=round_effort_to_slots(15),
                first_response_due_ts=datetime(2026, 3, 2, 8, 45, 0),
                resolution_due_ts=datetime(2026, 3, 2, 9, 15, 0),
            ),
            TicketRecord(
                ticket_id="TKT-D",
                arrival_ts=datetime(2026, 3, 2, 8, 15, 0),
                release_ts=datetime(2026, 3, 2, 8, 15, 0),
                queue="Product Support",
                priority="P2",
                language="EN",
                estimated_effort_min=45,
                duration_slots=round_effort_to_slots(45),
                first_response_due_ts=datetime(2026, 3, 2, 8, 45, 0),
                resolution_due_ts=datetime(2026, 3, 2, 10, 0, 0),
            ),
        ]

        myopic = run_greedy_baseline(tickets, agents)
        lookahead = run_lookahead_greedy(tickets, agents)

        self.assertGreaterEqual(
            lookahead.metrics["scheduled_tickets"], myopic.metrics["scheduled_tickets"]
        )
        self.assertEqual(lookahead.metrics["scheduled_tickets"], 3)
        self.assertEqual(myopic.metrics["scheduled_tickets"], 2)

    def test_from_csv_loader_path_matches_direct_run(self) -> None:
        from_csv = run_lookahead_greedy_from_csv(TICKETS_PATH, AGENTS_PATH)
        self.assertEqual(
            [entry.to_row() for entry in from_csv.schedule],
            [entry.to_row() for entry in self.schedule],
        )


if __name__ == "__main__":
    unittest.main()
