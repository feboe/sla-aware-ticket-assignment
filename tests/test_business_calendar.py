"""Tests for the shared SLA business-time semantics."""

import unittest
from datetime import datetime, time

from src.business_calendar import (
    add_business_minutes,
    business_minutes_between,
    business_slot_offset_floor,
)
from src.evaluation import ScheduleEntry, compute_schedule_metrics, tardiness_minutes
from src.preprocessing import AgentRecord


class TestBusinessCalendar(unittest.TestCase):
    """Keep generator, evaluator, and model time semantics aligned."""

    def test_night_boundary_does_not_advance_sla_clock(self) -> None:
        monday_close = datetime(2026, 3, 2, 16, 0)
        tuesday_open = datetime(2026, 3, 3, 8, 0)

        self.assertEqual(business_minutes_between(monday_close, tuesday_open), 0.0)
        self.assertEqual(tardiness_minutes(tuesday_open, monday_close), 0.0)

    def test_weekend_boundary_and_business_addition(self) -> None:
        friday_late = datetime(2026, 3, 6, 15, 30)
        monday_morning = datetime(2026, 3, 9, 9, 30)

        self.assertEqual(add_business_minutes(friday_late, 120), monday_morning)
        self.assertEqual(business_minutes_between(friday_late, monday_morning), 120.0)

    def test_twenty_four_business_hours_equal_three_business_days(self) -> None:
        monday_open = datetime(2026, 3, 2, 8, 0)
        wednesday_close = datetime(2026, 3, 4, 16, 0)

        self.assertEqual(add_business_minutes(monday_open, 24 * 60), wednesday_close)
        self.assertEqual(
            business_minutes_between(monday_open, wednesday_close),
            24 * 60,
        )

    def test_non_slot_aligned_times_use_exact_minutes_and_floor_slots(self) -> None:
        due_ts = datetime(2026, 3, 2, 15, 58)
        actual_ts = datetime(2026, 3, 3, 8, 3)

        self.assertEqual(business_minutes_between(due_ts, actual_ts), 5.0)
        self.assertEqual(tardiness_minutes(actual_ts, due_ts), 5.0)
        self.assertEqual(business_slot_offset_floor(actual_ts, due_ts, 5), 1)
        self.assertEqual(
            add_business_minutes(due_ts, 5),
            actual_ts,
        )

    def test_outside_window_is_clamped_when_measuring_business_time(self) -> None:
        self.assertEqual(
            business_minutes_between(
                datetime(2026, 3, 2, 17, 0), datetime(2026, 3, 3, 9, 0)
            ),
            60.0,
        )

    def test_scheduled_and_backlog_metrics_use_business_minutes(self) -> None:
        agent = AgentRecord(
            agent_id="AG-01",
            queue_permissions=frozenset({"Product Support"}),
            languages=frozenset({"EN"}),
            priority_scope=frozenset({"P1", "P2", "P3", "P4"}),
            shift_start=time(8, 0),
            shift_end=time(16, 0),
            capacity_min_per_day=480,
            scarce_resource=False,
        )
        due_ts = datetime(2026, 3, 2, 15, 45)
        completion_ts = datetime(2026, 3, 3, 8, 15)
        schedule = [
            ScheduleEntry(
                ticket_id="SCHEDULED",
                status="scheduled",
                agent_id="AG-01",
                start_ts=datetime(2026, 3, 3, 8, 0),
                completion_ts=completion_ts,
                arrival_ts=datetime(2026, 3, 2, 15, 0),
                queue="Product Support",
                priority="P1",
                language="EN",
                duration_slots=3,
                first_response_due_ts=due_ts,
                resolution_due_ts=due_ts,
                first_response_tardiness_min=tardiness_minutes(
                    datetime(2026, 3, 3, 8, 0), due_ts
                ),
                resolution_tardiness_min=tardiness_minutes(completion_ts, due_ts),
            ),
            ScheduleEntry(
                ticket_id="BACKLOG",
                status="backlog_end",
                agent_id="",
                start_ts=None,
                completion_ts=None,
                arrival_ts=datetime(2026, 3, 2, 15, 0),
                queue="Product Support",
                priority="P2",
                language="EN",
                duration_slots=3,
                first_response_due_ts=due_ts,
                resolution_due_ts=due_ts,
                first_response_tardiness_min=None,
                resolution_tardiness_min=None,
            ),
        ]

        metrics = compute_schedule_metrics(
            schedule,
            [agent],
            replay_day_count=2,
            workload_minutes_per_agent={"AG-01": 15},
            horizon_start_ts=datetime(2026, 3, 2, 8, 0),
            final_horizon_end=completion_ts,
        )

        self.assertEqual(metrics["scheduled"]["first_response_tardiness_min"], 15.0)
        self.assertEqual(metrics["scheduled"]["resolution_tardiness_min"], 30.0)
        self.assertEqual(metrics["backlog"]["first_response_tardiness_min"], 30.0)
        self.assertEqual(metrics["backlog"]["resolution_tardiness_min"], 30.0)

    def test_solver_uses_business_slot_offsets_across_a_weekend_gap(self) -> None:
        try:
            import ortools  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("OR-Tools is not installed in this environment.")

        from src.or_preparation import prepare_or_scheduler_instance
        from src.or_solver import solve_or_scheduler_instance
        from src.preprocessing import TicketRecord

        agent = AgentRecord(
            agent_id="AG-01",
            queue_permissions=frozenset({"Product Support"}),
            languages=frozenset({"DE"}),
            priority_scope=frozenset({"P1", "P2", "P3", "P4"}),
            shift_start=time(8, 0),
            shift_end=time(9, 0),
            capacity_min_per_day=60,
            scarce_resource=False,
        )
        ticket = TicketRecord(
            ticket_id="WEEKEND-DUE",
            arrival_ts=datetime(2026, 3, 6, 15, 0),
            release_ts=datetime(2026, 3, 6, 15, 0),
            queue="Product Support",
            priority="P2",
            language="EN",
            estimated_effort_min=15,
            duration_slots=3,
            first_response_due_ts=datetime(2026, 3, 6, 15, 55),
            resolution_due_ts=datetime(2026, 3, 6, 15, 55),
        )
        instance = prepare_or_scheduler_instance(
            None,
            None,
            datetime(2026, 3, 9, 8, 0),
            tickets_override=[ticket],
            agents_override=[agent],
        )
        artifacts = solve_or_scheduler_instance(instance, time_limit_sec=1, num_workers=1)

        self.assertEqual(instance.horizon_slot_count, 12)
        self.assertEqual(
            business_slot_offset_floor(ticket.first_response_due_ts, instance.decision_ts, 5),
            -1,
        )
        self.assertEqual(artifacts.status_name, "OPTIMAL")
        self.assertEqual(artifacts.objective_value, 2860.5)


if __name__ == "__main__":
    unittest.main()
