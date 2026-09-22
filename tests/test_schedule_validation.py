"""Focused contract tests for the shared final-output validation layer."""

import copy
import unittest
from dataclasses import replace
from datetime import datetime, time
from unittest.mock import patch

from scripts import run_greedy_baseline as greedy_cli
from src.evaluation import ScheduleEntry, compute_schedule_metrics
from src.greedy_baseline import run_greedy_baseline
from src.lookahead_greedy import run_lookahead_greedy
from src.or_scheduler import run_or_scheduler
from src.preprocessing import (
    AgentRecord,
    TicketRecord,
    derive_replay_horizon,
    load_agents,
    load_tickets,
)
from src.schedule_validation import (
    OutputValidationError,
    ValidationViolation,
    collect_output_violations,
    validate_output,
)


class TestScheduleValidation(unittest.TestCase):
    """Test each stable final-output validation rule independently."""

    def setUp(self) -> None:
        self.agent = AgentRecord(
            "AG-01",
            frozenset({"Product Support"}),
            frozenset({"EN"}),
            frozenset({"P1", "P2", "P3", "P4"}),
            time(8),
            time(9),
            60,
            False,
        )
        self.ticket = TicketRecord(
            "TKT-01",
            datetime(2026, 3, 2, 8),
            datetime(2026, 3, 2, 8),
            "Product Support",
            "P1",
            "EN",
            15,
            3,
            datetime(2026, 3, 2, 8, 15),
            datetime(2026, 3, 2, 8, 30),
        )
        self.entry = ScheduleEntry(
            self.ticket.ticket_id,
            "scheduled",
            self.agent.agent_id,
            datetime(2026, 3, 2, 8),
            datetime(2026, 3, 2, 8, 15),
            self.ticket.arrival_ts,
            self.ticket.queue,
            self.ticket.priority,
            self.ticket.language,
            self.ticket.duration_slots,
            self.ticket.first_response_due_ts,
            self.ticket.resolution_due_ts,
            0.0,
            0.0,
        )
        self.schedule = [self.entry]
        days, start, end = derive_replay_horizon([self.ticket], [self.agent])
        self.metrics = compute_schedule_metrics(
            self.schedule, [self.agent], len(days), {self.agent.agent_id: 15}, start, end
        )

    def codes(self, schedule=None, metrics=None, tickets=None, agents=None) -> set[str]:
        return {
            v.code
            for v in collect_output_violations(
                self.schedule if schedule is None else schedule,
                self.metrics if metrics is None else metrics,
                [self.ticket] if tickets is None else tickets,
                [self.agent] if agents is None else agents,
            )
        }

    def assert_code(self, expected: str, **kwargs) -> None:
        self.assertIn(expected, self.codes(**kwargs))

    def test_validates_all_current_policy_results(self) -> None:
        tickets, agents = load_tickets("data/tickets.csv"), load_agents("data/agents.csv")
        for result in (
            run_greedy_baseline(tickets, agents),
            run_lookahead_greedy(tickets, agents),
            run_or_scheduler(tickets, agents, time_limit_sec=1, num_workers=1),
        ):
            validate_output(result.schedule, result.metrics, tickets, agents)

    def test_missing_duplicate_and_unknown_ticket_rows(self) -> None:
        self.assert_code("MISSING_TICKET", schedule=[])
        self.assert_code("DUPLICATE_TICKET", schedule=self.schedule * 2)
        self.assert_code(
            "UNKNOWN_TICKET", schedule=[replace(self.entry, ticket_id="UNKNOWN")]
        )

    def test_final_status_is_restricted(self) -> None:
        self.assert_code(
            "INVALID_STATUS", schedule=[replace(self.entry, status="pending")]
        )

    def test_scheduled_row_requires_assignment_timing_and_tardiness_fields(self) -> None:
        entry = replace(
            self.entry,
            agent_id="",
            start_ts=None,
            completion_ts=None,
            first_response_tardiness_min=None,
            resolution_tardiness_min=None,
        )
        self.assert_code("SCHEDULED_REQUIRED_FIELD", schedule=[entry])

    def test_backlog_row_requires_assignment_and_timing_fields_to_be_empty(self) -> None:
        backlog = replace(
            self.entry,
            status="backlog_end",
            agent_id="",
            start_ts=None,
            completion_ts=None,
            first_response_tardiness_min=None,
            resolution_tardiness_min=None,
        )
        self.assertNotIn("BACKLOG_FIELD_NOT_EMPTY", self.codes(schedule=[backlog]))
        self.assert_code(
            "BACKLOG_FIELD_NOT_EMPTY", schedule=[replace(backlog, agent_id="AG-01")]
        )

    def test_ticket_master_data_must_be_unchanged(self) -> None:
        self.assert_code(
            "TICKET_FIELD_MISMATCH", schedule=[replace(self.entry, queue="Other")]
        )

    def test_unknown_agent_is_rejected(self) -> None:
        self.assert_code("UNKNOWN_AGENT", schedule=[replace(self.entry, agent_id="NOPE")])

    def test_ineligible_agent_is_rejected(self) -> None:
        ineligible = replace(self.agent, agent_id="AG-02", languages=frozenset({"DE"}))
        self.assert_code(
            "INELIGIBLE_AGENT",
            schedule=[replace(self.entry, agent_id="AG-02")],
            agents=[self.agent, ineligible],
        )

    def test_start_before_arrival_is_rejected(self) -> None:
        entry = replace(
            self.entry,
            start_ts=datetime(2026, 3, 2, 7, 55),
            completion_ts=datetime(2026, 3, 2, 8, 10),
        )
        self.assert_code("START_BEFORE_ARRIVAL", schedule=[entry])

    def test_start_before_release_is_rejected_separately(self) -> None:
        ticket = replace(self.ticket, release_ts=datetime(2026, 3, 2, 8, 10))
        entry = replace(
            self.entry,
            start_ts=datetime(2026, 3, 2, 8, 5),
            completion_ts=datetime(2026, 3, 2, 8, 20),
        )
        codes = self.codes(schedule=[entry], tickets=[ticket])
        self.assertIn("START_BEFORE_RELEASE", codes)
        self.assertNotIn("START_BEFORE_ARRIVAL", codes)

    def test_start_must_align_to_a_five_minute_slot(self) -> None:
        entry = replace(
            self.entry,
            start_ts=datetime(2026, 3, 2, 8, 1),
            completion_ts=datetime(2026, 3, 2, 8, 16),
        )
        self.assert_code("START_NOT_ON_SLOT", schedule=[entry])

    def test_duration_must_match_ticket_slots(self) -> None:
        self.assert_code(
            "INVALID_DURATION",
            schedule=[replace(self.entry, completion_ts=datetime(2026, 3, 2, 8, 20))],
        )

    def test_work_must_finish_on_the_start_day(self) -> None:
        ticket = replace(self.ticket, duration_slots=192, estimated_effort_min=960)
        entry = replace(
            self.entry, completion_ts=datetime(2026, 3, 3), duration_slots=192
        )
        self.assert_code("CROSS_DAY_COMPLETION", schedule=[entry], tickets=[ticket])

    def test_work_must_fit_assigned_shift(self) -> None:
        entry = replace(
            self.entry,
            start_ts=datetime(2026, 3, 2, 8, 50),
            completion_ts=datetime(2026, 3, 2, 9, 5),
        )
        self.assert_code("OUTSIDE_SHIFT", schedule=[entry])

    def test_business_tardiness_must_match_schedule(self) -> None:
        self.assert_code(
            "INVALID_TARDINESS",
            schedule=[replace(self.entry, first_response_tardiness_min=5.0)],
        )

    def test_nonfinite_ticket_tardiness_is_rejected_without_metric_recompute_crash(
        self,
    ) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            self.assert_code(
                "INVALID_TARDINESS",
                schedule=[replace(self.entry, first_response_tardiness_min=value)],
            )

    def test_agent_overlap_is_rejected(self) -> None:
        ticket = replace(self.ticket, ticket_id="TKT-02")
        self.assert_code(
            "AGENT_OVERLAP",
            schedule=[self.entry, replace(self.entry, ticket_id="TKT-02")],
            tickets=[self.ticket, ticket],
        )

    def test_daily_agent_capacity_is_rejected(self) -> None:
        ticket = replace(self.ticket, ticket_id="TKT-02")
        entry = replace(
            self.entry,
            ticket_id="TKT-02",
            start_ts=datetime(2026, 3, 2, 8, 15),
            completion_ts=datetime(2026, 3, 2, 8, 30),
        )
        self.assert_code(
            "CAPACITY_EXCEEDED",
            schedule=[self.entry, entry],
            tickets=[self.ticket, ticket],
            agents=[replace(self.agent, capacity_min_per_day=15)],
        )

    def test_top_level_metrics_are_recomputed(self) -> None:
        metrics = copy.deepcopy(self.metrics)
        metrics["total_tickets"] = 99
        self.assert_code("METRIC_MISMATCH", metrics=metrics)

    def test_priority_metrics_are_recomputed(self) -> None:
        metrics = copy.deepcopy(self.metrics)
        metrics["scheduled"]["by_priority"]["P1"]["ticket_count"] = 2
        self.assert_code("METRIC_MISMATCH", metrics=metrics)

    def test_agent_metrics_are_recomputed(self) -> None:
        metrics = copy.deepcopy(self.metrics)
        metrics["agent_utilization"]["AG-01"]["workload_minutes"] = 0
        self.assert_code("METRIC_MISMATCH", metrics=metrics)

    def test_nonfinite_metrics_are_rejected(self) -> None:
        metrics = copy.deepcopy(self.metrics)
        metrics["overall_agent_utilization"]["utilization"] = float("nan")
        self.assert_code("METRIC_MISMATCH", metrics=metrics)

    def test_or_diagnostics_are_allowed_as_metrics_extras(self) -> None:
        metrics = copy.deepcopy(self.metrics)
        metrics["solver_status_counts"] = {"OPTIMAL": 1}
        metrics["avg_solve_time_sec"] = 0.123
        self.assertEqual(self.codes(metrics=metrics), set())

    def test_multiple_violations_have_deterministic_order(self) -> None:
        ticket_two = replace(self.ticket, ticket_id="TKT-02")
        violations = collect_output_violations(
            [], self.metrics, [ticket_two, self.ticket], [self.agent]
        )
        self.assertEqual(len(violations), 2)
        self.assertEqual(violations, sorted(violations, key=lambda item: item.sort_key()))
        with self.assertRaises(OutputValidationError) as caught:
            validate_output([], self.metrics, [ticket_two, self.ticket], [self.agent])
        self.assertEqual(list(caught.exception.violations), violations)

    def test_each_runner_propagates_validation_failure_before_returning(self) -> None:
        failure = OutputValidationError([ValidationViolation("TEST_FAILURE", "forced")])
        for target, runner, kwargs in (
            ("src.greedy_baseline.validate_output", run_greedy_baseline, {}),
            ("src.lookahead_greedy.validate_output", run_lookahead_greedy, {}),
            (
                "src.or_scheduler.validate_output",
                run_or_scheduler,
                {"time_limit_sec": 1, "num_workers": 1},
            ),
        ):
            with patch(target, side_effect=failure) as validate:
                with self.assertRaises(OutputValidationError):
                    runner([self.ticket], [self.agent], **kwargs)
                validate.assert_called_once()

    def test_greedy_cli_does_not_write_after_runner_validation_failure(self) -> None:
        failure = OutputValidationError([ValidationViolation("TEST_FAILURE", "forced")])
        with patch(
            "scripts.run_greedy_baseline.run_greedy_baseline_from_csv",
            side_effect=failure,
        ):
            with patch("scripts.run_greedy_baseline.write_baseline_outputs") as writer:
                with patch("sys.argv", ["run_greedy_baseline.py"]):
                    with self.assertRaises(OutputValidationError):
                        greedy_cli.main()
                writer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
