"""Solver-independent validation for final ticket-assignment outputs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from numbers import Real
from typing import Any

from src.evaluation import ScheduleEntry, compute_schedule_metrics, tardiness_minutes
from src.preprocessing import (
    AgentRecord,
    SLOT_MINUTES,
    TicketRecord,
    combine_date_and_time,
    derive_replay_horizon,
    is_agent_feasible,
)

METRIC_TOLERANCE = 0.01


def _is_finite_number(value: object) -> bool:
    """Return whether value is a finite, non-boolean real number."""

    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class ValidationViolation:
    """One deterministic, machine-readable output contract violation."""

    code: str
    message: str
    ticket_id: str | None = None
    agent_id: str | None = None

    def sort_key(self) -> tuple[str, str, str, str]:
        return (self.code, self.ticket_id or "", self.agent_id or "", self.message)

    def __str__(self) -> str:
        context = []
        if self.ticket_id is not None:
            context.append(f"ticket_id={self.ticket_id}")
        if self.agent_id is not None:
            context.append(f"agent_id={self.agent_id}")
        suffix = f" ({', '.join(context)})" if context else ""
        return f"{self.code}: {self.message}{suffix}"


class OutputValidationError(ValueError):
    """Raised when a final schedule or its metrics violate the output contract."""

    def __init__(self, violations: list[ValidationViolation]):
        self.violations = tuple(sorted(violations, key=ValidationViolation.sort_key))
        super().__init__(
            "Output validation failed:\n" + "\n".join(map(str, self.violations))
        )


def _violation(
    code: str,
    message: str,
    entry: ScheduleEntry | None = None,
    agent_id: str | None = None,
) -> ValidationViolation:
    return ValidationViolation(
        code=code,
        message=message,
        ticket_id=entry.ticket_id if entry else None,
        agent_id=(
            agent_id if agent_id is not None else (entry.agent_id if entry else None)
        ),
    )


def _is_slot_aligned(timestamp: datetime) -> bool:
    return (
        timestamp.minute % SLOT_MINUTES == 0
        and timestamp.second == 0
        and timestamp.microsecond == 0
    )


def _metric_violations(
    expected: Any, observed: Any, path: str = "metrics"
) -> list[ValidationViolation]:
    """Compare only expected core metric fields; policy diagnostics are extras."""

    violations: list[ValidationViolation] = []
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            return [ValidationViolation("METRIC_MISMATCH", f"{path} must be an object.")]
        for key, value in expected.items():
            if key not in observed:
                violations.append(
                    ValidationViolation(
                        "METRIC_MISMATCH", f"Missing metric field {path}.{key}."
                    )
                )
            else:
                violations.extend(
                    _metric_violations(value, observed[key], f"{path}.{key}")
                )
        return violations
    if _is_finite_number(expected):
        if not _is_finite_number(observed):
            return [ValidationViolation("METRIC_MISMATCH", f"{path} must be numeric.")]
        if abs(float(expected) - float(observed)) > METRIC_TOLERANCE:
            return [
                ValidationViolation(
                    "METRIC_MISMATCH",
                    f"{path} is {observed!r}; expected {expected!r} within {METRIC_TOLERANCE}.",
                )
            ]
        return []
    if expected != observed:
        return [
            ValidationViolation(
                "METRIC_MISMATCH", f"{path} is {observed!r}; expected {expected!r}."
            )
        ]
    return []


def _can_recompute_metrics(
    schedule: list[ScheduleEntry], tickets: list[TicketRecord], agents: list[AgentRecord]
) -> bool:
    """Avoid masking collected schedule failures with aggregation exceptions."""

    ticket_by_id = {ticket.ticket_id: ticket for ticket in tickets}
    agent_ids = {agent.agent_id for agent in agents}
    if len(schedule) != len(tickets) or len(
        {entry.ticket_id for entry in schedule}
    ) != len(schedule):
        return False
    for entry in schedule:
        ticket = ticket_by_id.get(entry.ticket_id)
        if ticket is None or entry.status not in {"scheduled", "backlog_end"}:
            return False
        if (
            entry.arrival_ts != ticket.arrival_ts
            or entry.queue != ticket.queue
            or entry.priority != ticket.priority
            or entry.language != ticket.language
            or entry.duration_slots != ticket.duration_slots
            or entry.first_response_due_ts != ticket.first_response_due_ts
            or entry.resolution_due_ts != ticket.resolution_due_ts
        ):
            return False
        if entry.status == "scheduled" and entry.agent_id not in agent_ids:
            return False
        if entry.status == "scheduled" and not (
            _is_finite_number(entry.first_response_tardiness_min)
            and _is_finite_number(entry.resolution_tardiness_min)
        ):
            return False
    return set(ticket_by_id) == {entry.ticket_id for entry in schedule}


def collect_output_violations(
    schedule: list[ScheduleEntry],
    metrics: dict[str, Any],
    tickets: list[TicketRecord],
    agents: list[AgentRecord],
) -> list[ValidationViolation]:
    """Collect every final-output violation without raising on the first one."""

    violations: list[ValidationViolation] = []
    ticket_by_id = {ticket.ticket_id: ticket for ticket in tickets}
    agent_by_id = {agent.agent_id: agent for agent in agents}
    seen_ids: set[str] = set()
    scheduled_entries: list[ScheduleEntry] = []

    for entry in schedule:
        ticket = ticket_by_id.get(entry.ticket_id)
        if entry.ticket_id in seen_ids:
            violations.append(
                _violation("DUPLICATE_TICKET", "Ticket appears more than once.", entry)
            )
        seen_ids.add(entry.ticket_id)
        if ticket is None:
            violations.append(
                _violation(
                    "UNKNOWN_TICKET", "Ticket is not present in input tickets.", entry
                )
            )
            continue
        if entry.status not in {"scheduled", "backlog_end"}:
            violations.append(
                _violation(
                    "INVALID_STATUS",
                    "Final status must be scheduled or backlog_end.",
                    entry,
                )
            )
            continue

        fields_match = (
            entry.arrival_ts == ticket.arrival_ts
            and entry.queue == ticket.queue
            and entry.priority == ticket.priority
            and entry.language == ticket.language
            and entry.duration_slots == ticket.duration_slots
            and entry.first_response_due_ts == ticket.first_response_due_ts
            and entry.resolution_due_ts == ticket.resolution_due_ts
        )
        if not fields_match:
            violations.append(
                _violation(
                    "TICKET_FIELD_MISMATCH",
                    "Ticket master data differs from input.",
                    entry,
                )
            )

        if entry.status == "backlog_end":
            if any(
                value not in (None, "")
                for value in (
                    entry.agent_id,
                    entry.start_ts,
                    entry.completion_ts,
                    entry.first_response_tardiness_min,
                    entry.resolution_tardiness_min,
                )
            ):
                violations.append(
                    _violation(
                        "BACKLOG_FIELD_NOT_EMPTY",
                        "Backlog rows must not contain assignment or timing fields.",
                        entry,
                    )
                )
            continue

        scheduled_entries.append(entry)
        has_timing_fields = isinstance(entry.start_ts, datetime) and isinstance(
            entry.completion_ts, datetime
        )
        if not entry.agent_id or not has_timing_fields:
            violations.append(
                _violation(
                    "SCHEDULED_REQUIRED_FIELD",
                    "Scheduled rows require agent_id, start_ts, and completion_ts.",
                    entry,
                )
            )
        if (
            entry.first_response_tardiness_min is None
            or entry.resolution_tardiness_min is None
        ):
            violations.append(
                _violation(
                    "SCHEDULED_REQUIRED_FIELD",
                    "Scheduled rows require both tardiness values.",
                    entry,
                )
            )

        agent = agent_by_id.get(entry.agent_id)
        if agent is None:
            violations.append(
                _violation(
                    "UNKNOWN_AGENT",
                    "Assigned agent is not present in input agents.",
                    entry,
                )
            )
        elif not is_agent_feasible(ticket, agent):
            violations.append(
                _violation(
                    "INELIGIBLE_AGENT", "Assigned agent cannot serve this ticket.", entry
                )
            )

        if not has_timing_fields:
            continue

        if entry.start_ts < ticket.arrival_ts:
            violations.append(
                _violation(
                    "START_BEFORE_ARRIVAL", "Start precedes arrival timestamp.", entry
                )
            )
        if entry.start_ts < ticket.release_ts:
            violations.append(
                _violation(
                    "START_BEFORE_RELEASE", "Start precedes release timestamp.", entry
                )
            )
        if not _is_slot_aligned(entry.start_ts):
            violations.append(
                _violation(
                    "START_NOT_ON_SLOT", "Start must align to a 5-minute slot.", entry
                )
            )
        expected_completion = entry.start_ts + timedelta(
            minutes=ticket.duration_slots * SLOT_MINUTES
        )
        if entry.completion_ts != expected_completion:
            violations.append(
                _violation(
                    "INVALID_DURATION",
                    "Completion does not match ticket slot duration.",
                    entry,
                )
            )
        if entry.completion_ts.date() != entry.start_ts.date():
            violations.append(
                _violation(
                    "CROSS_DAY_COMPLETION",
                    "Scheduled work must complete on its start day.",
                    entry,
                )
            )
        if agent is not None:
            shift_start = combine_date_and_time(entry.start_ts.date(), agent.shift_start)
            shift_end = combine_date_and_time(entry.start_ts.date(), agent.shift_end)
            if entry.start_ts < shift_start or entry.completion_ts > shift_end:
                violations.append(
                    _violation(
                        "OUTSIDE_SHIFT",
                        "Scheduled work falls outside the assigned agent shift.",
                        entry,
                    )
                )
        if entry.first_response_tardiness_min is not None:
            expected_tardiness = tardiness_minutes(
                entry.start_ts, ticket.first_response_due_ts
            )
            if (
                not _is_finite_number(entry.first_response_tardiness_min)
                or abs(entry.first_response_tardiness_min - expected_tardiness)
                > METRIC_TOLERANCE
            ):
                violations.append(
                    _violation(
                        "INVALID_TARDINESS",
                        "First-response tardiness is incorrect.",
                        entry,
                    )
                )
        if entry.resolution_tardiness_min is not None:
            expected_tardiness = tardiness_minutes(
                entry.completion_ts, ticket.resolution_due_ts
            )
            if (
                not _is_finite_number(entry.resolution_tardiness_min)
                or abs(entry.resolution_tardiness_min - expected_tardiness)
                > METRIC_TOLERANCE
            ):
                violations.append(
                    _violation(
                        "INVALID_TARDINESS", "Resolution tardiness is incorrect.", entry
                    )
                )

    for ticket_id in sorted(set(ticket_by_id) - seen_ids):
        violations.append(
            ValidationViolation(
                "MISSING_TICKET", "Ticket has no final schedule row.", ticket_id=ticket_id
            )
        )

    by_agent_day: dict[tuple[str, object], list[ScheduleEntry]] = defaultdict(list)
    for entry in scheduled_entries:
        if (
            entry.agent_id in agent_by_id
            and isinstance(entry.start_ts, datetime)
            and isinstance(entry.completion_ts, datetime)
        ):
            by_agent_day[(entry.agent_id, entry.start_ts.date())].append(entry)
    for (agent_id, _day), entries in by_agent_day.items():
        agent = agent_by_id[agent_id]
        ordered = sorted(
            entries, key=lambda item: (item.start_ts, item.completion_ts, item.ticket_id)
        )
        latest_completion: datetime | None = None
        workload = 0
        for entry in ordered:
            if latest_completion is not None and entry.start_ts < latest_completion:
                violations.append(
                    _violation(
                        "AGENT_OVERLAP",
                        "Assigned work overlaps another ticket.",
                        entry,
                        agent_id,
                    )
                )
            if latest_completion is None or entry.completion_ts > latest_completion:
                latest_completion = entry.completion_ts
            workload += entry.duration_slots * SLOT_MINUTES
        if workload > agent.capacity_min_per_day:
            violations.append(
                ValidationViolation(
                    "CAPACITY_EXCEEDED",
                    f"Daily workload {workload} exceeds capacity {agent.capacity_min_per_day}.",
                    agent_id=agent_id,
                )
            )

    if _can_recompute_metrics(schedule, tickets, agents):
        replay_days, horizon_start, horizon_end = derive_replay_horizon(tickets, agents)
        workload_by_agent = {agent.agent_id: 0 for agent in agents}
        for entry in schedule:
            if entry.status == "scheduled":
                workload_by_agent[entry.agent_id] += entry.duration_slots * SLOT_MINUTES
        expected_metrics = compute_schedule_metrics(
            schedule,
            agents,
            len(replay_days),
            workload_by_agent,
            horizon_start,
            horizon_end,
        )
        violations.extend(_metric_violations(expected_metrics, metrics))

    return sorted(violations, key=ValidationViolation.sort_key)


def validate_output(
    schedule: list[ScheduleEntry],
    metrics: dict[str, Any],
    tickets: list[TicketRecord],
    agents: list[AgentRecord],
) -> None:
    """Raise one aggregated error if a final policy output is invalid."""

    violations = collect_output_violations(schedule, metrics, tickets, agents)
    if violations:
        raise OutputValidationError(violations)
