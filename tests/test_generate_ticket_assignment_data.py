"""Contract tests for the synthetic ticket-demand dataset generator."""

from __future__ import annotations

import unittest
from collections import Counter
from datetime import datetime

from scripts.generate_ticket_assignment_data import (
    COMPLEXITY_BY_QUEUE,
    EFFORT_RANGES,
    LANGUAGE_WEIGHTS,
    LANGUAGES,
    QUEUE_SKILL,
    QUEUES,
    QUEUE_WEIGHTS,
    SEED,
    SHIFT_DURATION_HOURS,
    SHIFT_START_HOUR,
    TIMESTAMP_FORMAT,
    add_business_minutes,
    generate_dataset,
    validate_dataset,
)


EXPECTED_COLUMNS = [
    "ticket_id",
    "arrival_ts",
    "queue",
    "priority",
    "language",
    "channel",
    "customer_tier",
    "is_vip",
    "is_reopened",
    "complexity",
    "estimated_effort_min",
    "first_response_sla_min",
    "first_response_due_ts",
    "resolution_sla_business_min",
    "resolution_due_ts",
    "required_skill_tags",
    "ticket_subject",
]


def feasible_efforts(queue: str, complexity: str, priority: str, is_vip: int) -> set[int]:
    """Rebuild the reachable effort values for one queue/complexity combination."""

    lo, hi = EFFORT_RANGES[(queue, complexity)]
    values = set()
    for base_effort in range(lo, hi + 1):
        effort = base_effort
        if priority == "P1":
            effort = int(round(effort * 1.15))
        elif priority == "P2":
            effort = int(round(effort * 1.05))
        if is_vip and queue in {"Product Support", "Integrations/API"}:
            effort = int(round(effort * 1.05))
        values.add(max(5, effort))
    return values


class TestGenerateTicketAssignmentData(unittest.TestCase):
    """Validate schema, calendar logic, and sanity checks for ticket generation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = generate_dataset(SEED)
        validate_dataset(cls.rows)

    def test_seed_is_deterministic(self) -> None:
        second_run = generate_dataset(SEED)
        self.assertEqual(
            [row.__dict__ for row in self.rows],
            [row.__dict__ for row in second_run],
        )

    def test_schema_matches_public_contract(self) -> None:
        self.assertEqual(list(self.rows[0].__dict__.keys()), EXPECTED_COLUMNS)

    def test_deadlines_are_monotonic(self) -> None:
        previous_arrival = None
        for row in self.rows:
            arrival = datetime.strptime(row.arrival_ts, TIMESTAMP_FORMAT)
            first_response_due = datetime.strptime(
                row.first_response_due_ts, TIMESTAMP_FORMAT
            )
            resolution_due = datetime.strptime(row.resolution_due_ts, TIMESTAMP_FORMAT)

            if previous_arrival is not None:
                self.assertLessEqual(previous_arrival, arrival)

            self.assertLessEqual(arrival, first_response_due)
            self.assertLessEqual(first_response_due, resolution_due)
            previous_arrival = arrival

    def test_business_calendar_rollover(self) -> None:
        friday_late = datetime(2026, 3, 6, 15, 30, 0)
        monday_morning = add_business_minutes(friday_late, 120)
        self.assertEqual(monday_morning, datetime(2026, 3, 9, 9, 30, 0))

        monday_late = datetime(2026, 3, 2, 15, 45, 0)
        next_day = add_business_minutes(monday_late, 30)
        self.assertEqual(next_day, datetime(2026, 3, 3, 8, 15, 0))

    def test_distribution_sanity_for_stable_inputs(self) -> None:
        row_count = len(self.rows)

        queue_counts = Counter(row.queue for row in self.rows)
        observed_queue_share = {
            queue: count / row_count for queue, count in queue_counts.items()
        }
        expected_queue_share = dict(zip(QUEUES, QUEUE_WEIGHTS))
        for queue, expected_share in expected_queue_share.items():
            self.assertAlmostEqual(
                observed_queue_share[queue], expected_share, delta=0.06
            )

        language_counts = Counter(row.language for row in self.rows)
        observed_language_share = {
            language: count / row_count for language, count in language_counts.items()
        }
        expected_language_share = dict(zip(LANGUAGES, LANGUAGE_WEIGHTS))
        for language, expected_share in expected_language_share.items():
            self.assertAlmostEqual(
                observed_language_share[language], expected_share, delta=0.05
            )

        for queue, expected_weights in COMPLEXITY_BY_QUEUE.items():
            queue_rows = [row for row in self.rows if row.queue == queue]
            complexity_counts = Counter(row.complexity for row in queue_rows)
            observed_shares = {
                complexity: complexity_counts[complexity] / len(queue_rows)
                for complexity in ("simple", "standard", "complex")
            }
            for complexity, expected_share in zip(
                ("simple", "standard", "complex"), expected_weights
            ):
                self.assertAlmostEqual(
                    observed_shares[complexity], expected_share, delta=0.18
                )

    def test_effort_values_are_plausible(self) -> None:
        for row in self.rows:
            valid_efforts = feasible_efforts(
                row.queue, row.complexity, row.priority, row.is_vip
            )
            self.assertIn(row.estimated_effort_min, valid_efforts)
            self.assertGreater(row.estimated_effort_min, 0)

    def test_required_skill_tags_follow_refined_rules(self) -> None:
        for row in self.rows:
            tags = set(row.required_skill_tags.split("|"))

            self.assertIn(QUEUE_SKILL[row.queue], tags)
            self.assertNotIn("senior_review", tags)

            if row.priority == "P1":
                self.assertIn("incident_escalation", tags)
            else:
                self.assertNotIn("incident_escalation", tags)

            if row.queue == "Product Support" and row.complexity == "complex":
                self.assertIn("product_specialist", tags)
            else:
                self.assertNotIn("product_specialist", tags)

            if row.queue == "Integrations/API" and row.complexity == "complex":
                self.assertIn("integration_specialist", tags)
            else:
                self.assertNotIn("integration_specialist", tags)

            if row.queue == "Integrations/API":
                self.assertIn("api_specialist", tags)
            else:
                self.assertNotIn("api_specialist", tags)

            if row.language == "DE":
                self.assertIn("de_language", tags)
            else:
                self.assertNotIn("de_language", tags)

            if row.customer_tier == "Enterprise" or row.is_vip:
                self.assertIn("enterprise_handling", tags)
            else:
                self.assertNotIn("enterprise_handling", tags)

    def test_all_timestamps_stay_inside_business_hours(self) -> None:
        for row in self.rows:
            for value in (
                row.arrival_ts,
                row.first_response_due_ts,
                row.resolution_due_ts,
            ):
                ts = datetime.strptime(value, TIMESTAMP_FORMAT)
                day_start = ts.replace(
                    hour=SHIFT_START_HOUR, minute=0, second=0, microsecond=0
                )
                day_end = day_start.replace(hour=SHIFT_START_HOUR + SHIFT_DURATION_HOURS)
                self.assertLess(ts.weekday(), 5)
                self.assertGreaterEqual(ts, day_start)
                self.assertLessEqual(ts, day_end)


if __name__ == "__main__":
    unittest.main()
