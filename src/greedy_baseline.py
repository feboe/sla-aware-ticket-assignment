"""Greedy baseline for the synthetic ticket-assignment problem.

This module implements a simple online dispatch policy that replays the synthetic support
environment in 15-minute slots. Trade-Offs are the following: tickets may only start
"now", there is no look-ahead, and hard feasibility is limited to queue, language, and
priority scope.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from scripts.generate_ticket_assignment_data import TIMESTAMP_FORMAT

SLOT_MINUTES = 15
PRIORITY_RANK = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
SCHEDULE_FIELDNAMES = [
    "ticket_id",
    "status",
    "agent_id",
    "start_ts",
    "completion_ts",
    "arrival_ts",
    "queue",
    "priority",
    "language",
    "duration_slots",
    "first_response_due_ts",
    "resolution_due_ts",
    "first_response_tardiness_min",
    "resolution_tardiness_min",
]


@dataclass(frozen=True)
class TicketRecord:
    """Demand-side ticket data after timestamp parsing and slot discretization."""

    ticket_id: str
    arrival_ts: datetime
    release_ts: datetime
    queue: str
    priority: str
    language: str
    estimated_effort_min: int
    duration_slots: int
    first_response_due_ts: datetime
    resolution_due_ts: datetime


@dataclass(frozen=True)
class AgentRecord:
    """Supply-side agent capabilities used by the greedy dispatcher."""

    agent_id: str
    queue_permissions: frozenset[str]
    languages: frozenset[str]
    priority_scope: frozenset[str]
    shift_start: time
    shift_end: time
    capacity_min_per_day: int
    scarce_resource: bool


@dataclass(frozen=True)
class ScheduleEntry:
    """Final schedule row for one ticket, either scheduled or left in backlog."""

    ticket_id: str
    status: str
    agent_id: str
    start_ts: datetime | None
    completion_ts: datetime | None
    arrival_ts: datetime
    queue: str
    priority: str
    language: str
    duration_slots: int
    first_response_due_ts: datetime
    resolution_due_ts: datetime
    first_response_tardiness_min: float | None
    resolution_tardiness_min: float | None

    def to_row(self) -> dict[str, str]:
        """Convert the schedule entry into a CSV-ready string dictionary."""

        return {
            "ticket_id": self.ticket_id,
            "status": self.status,
            "agent_id": self.agent_id,
            "start_ts": format_timestamp(self.start_ts),
            "completion_ts": format_timestamp(self.completion_ts),
            "arrival_ts": format_timestamp(self.arrival_ts),
            "queue": self.queue,
            "priority": self.priority,
            "language": self.language,
            "duration_slots": str(self.duration_slots),
            "first_response_due_ts": format_timestamp(self.first_response_due_ts),
            "resolution_due_ts": format_timestamp(self.resolution_due_ts),
            "first_response_tardiness_min": format_metric_value(
                self.first_response_tardiness_min
            ),
            "resolution_tardiness_min": format_metric_value(
                self.resolution_tardiness_min
            ),
        }


@dataclass(frozen=True)
class BaselineResult:
    """Container for the ticket-level schedule and aggregate baseline metrics."""

    schedule: list[ScheduleEntry]
    metrics: dict[str, Any]


def parse_timestamp(value: str) -> datetime:
    """Parse timestamps using the shared dataset format."""

    return datetime.strptime(value, TIMESTAMP_FORMAT)


def format_timestamp(value: datetime | None) -> str:
    """Serialize datetimes back into the shared dataset format."""

    if value is None:
        return ""
    return value.strftime(TIMESTAMP_FORMAT)


def format_metric_value(value: float | None) -> str:
    """Format numeric metrics for CSV output while preserving empty backlog rows."""

    if value is None:
        return ""
    return f"{value:.2f}"


def parse_pipe_set(value: str) -> frozenset[str]:
    """Parse pipe-delimited capability fields from the input CSVs."""

    return frozenset(part for part in value.split("|") if part)


def ceil_to_slot(ts: datetime) -> datetime:
    """Round a timestamp up to the next 15-minute decision slot."""

    slot_floor = ts.replace(
        minute=(ts.minute // SLOT_MINUTES) * SLOT_MINUTES,
        second=0,
        microsecond=0,
    )
    if ts == slot_floor:
        return slot_floor
    return slot_floor + timedelta(minutes=SLOT_MINUTES)


def round_effort_to_slots(estimated_effort_min: int) -> int:
    """Convert effort minutes into a non-zero number of 15-minute slots."""

    return max(1, math.ceil(estimated_effort_min / SLOT_MINUTES))


def tardiness_minutes(actual_ts: datetime, due_ts: datetime) -> float:
    """Return positive lateness in minutes and clip early completions to zero."""

    return round(max(0.0, (actual_ts - due_ts).total_seconds() / 60.0), 2)


def empty_priority_tardiness_metrics() -> dict[str, float | int]:
    """Build one empty backlog-metric bucket for a single priority class."""

    return {
        "ticket_count": 0,
        "effort_min": 0,
        "first_response_tardiness_min": 0.0,
        "resolution_tardiness_min": 0.0,
        "overdue_first_response_count": 0,
        "overdue_resolution_count": 0,
    }


def empty_metric_section() -> dict[str, Any]:
    """Build one empty metric section shared by scheduled and backlog views."""

    return {
        "ticket_count": 0,
        "effort_min": 0,
        "first_response_tardiness_min": 0.0,
        "resolution_tardiness_min": 0.0,
        "overdue_first_response_count": 0,
        "overdue_resolution_count": 0,
        "by_priority": {
            priority: empty_priority_tardiness_metrics() for priority in PRIORITY_RANK
        },
    }


def accumulate_metric_section(
    section: dict[str, Any],
    priority: str,
    effort_min: int,
    first_response_tardiness: float,
    resolution_tardiness: float,
) -> None:
    """Accumulate one ticket's contribution into a metric section."""

    priority_bucket = section["by_priority"][priority]

    section["ticket_count"] += 1
    section["effort_min"] += effort_min
    section["first_response_tardiness_min"] += first_response_tardiness
    section["resolution_tardiness_min"] += resolution_tardiness
    priority_bucket["ticket_count"] += 1
    priority_bucket["effort_min"] += effort_min
    priority_bucket["first_response_tardiness_min"] += first_response_tardiness
    priority_bucket["resolution_tardiness_min"] += resolution_tardiness

    if first_response_tardiness > 0:
        section["overdue_first_response_count"] += 1
        priority_bucket["overdue_first_response_count"] += 1
    if resolution_tardiness > 0:
        section["overdue_resolution_count"] += 1
        priority_bucket["overdue_resolution_count"] += 1


def finalize_metric_section(section: dict[str, Any]) -> dict[str, Any]:
    """Round float values while keeping the shared metric schema stable."""

    return {
        "ticket_count": section["ticket_count"],
        "effort_min": section["effort_min"],
        "first_response_tardiness_min": round(section["first_response_tardiness_min"], 2),
        "resolution_tardiness_min": round(section["resolution_tardiness_min"], 2),
        "overdue_first_response_count": section["overdue_first_response_count"],
        "overdue_resolution_count": section["overdue_resolution_count"],
        "by_priority": {
            priority: {
                "ticket_count": values["ticket_count"],
                "effort_min": values["effort_min"],
                "first_response_tardiness_min": round(
                    values["first_response_tardiness_min"], 2
                ),
                "resolution_tardiness_min": round(values["resolution_tardiness_min"], 2),
                "overdue_first_response_count": values["overdue_first_response_count"],
                "overdue_resolution_count": values["overdue_resolution_count"],
            }
            for priority, values in section["by_priority"].items()
        },
    }


def combine_date_and_time(day: date, value: time) -> datetime:
    """Create a datetime for a given business day and clock time."""

    return datetime.combine(day, value)


def business_days_inclusive(start_day: date, end_day: date) -> list[date]:
    """List weekdays in the closed interval from start_day to end_day."""

    days: list[date] = []
    current = start_day
    while current <= end_day:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def day_slot_starts(
    day: date, earliest_shift_start: time, latest_shift_end: time
) -> list[datetime]:
    """Enumerate 15-minute slot starts across the active business window."""

    slot_starts: list[datetime] = []
    current = combine_date_and_time(day, earliest_shift_start)
    day_end = combine_date_and_time(day, latest_shift_end)
    while current < day_end:
        slot_starts.append(current)
        current += timedelta(minutes=SLOT_MINUTES)
    return slot_starts


def load_tickets(ticket_csv_path: str | Path) -> list[TicketRecord]:
    """Load the ticket CSV and derive release times and slot durations."""

    path = Path(ticket_csv_path)
    tickets: list[TicketRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            arrival_ts = parse_timestamp(row["arrival_ts"])
            estimated_effort_min = int(row["estimated_effort_min"])
            tickets.append(
                TicketRecord(
                    ticket_id=row["ticket_id"],
                    arrival_ts=arrival_ts,
                    release_ts=ceil_to_slot(arrival_ts),
                    queue=row["queue"],
                    priority=row["priority"],
                    language=row["language"],
                    estimated_effort_min=estimated_effort_min,
                    duration_slots=round_effort_to_slots(estimated_effort_min),
                    first_response_due_ts=parse_timestamp(row["first_response_due_ts"]),
                    resolution_due_ts=parse_timestamp(row["resolution_due_ts"]),
                )
            )
    return sorted(tickets, key=lambda ticket: (ticket.arrival_ts, ticket.ticket_id))


def load_agents(agent_csv_path: str | Path) -> list[AgentRecord]:
    """Load the fixed agent roster and parse capability sets."""

    path = Path(agent_csv_path)
    agents: list[AgentRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            agents.append(
                AgentRecord(
                    agent_id=row["agent_id"],
                    queue_permissions=parse_pipe_set(row["queue_permissions"]),
                    languages=parse_pipe_set(row["languages"]),
                    priority_scope=parse_pipe_set(row["priority_scope"]),
                    shift_start=datetime.strptime(row["shift_start"], "%H:%M").time(),
                    shift_end=datetime.strptime(row["shift_end"], "%H:%M").time(),
                    capacity_min_per_day=int(row["capacity_min_per_day"]),
                    scarce_resource=row["scarce_resource"] == "1",
                )
            )
    return sorted(agents, key=lambda agent: agent.agent_id)


def ticket_sort_key(ticket: TicketRecord) -> tuple[Any, ...]:
    """Return the deterministic urgency order used by the greedy policy."""

    return (
        PRIORITY_RANK[ticket.priority],
        ticket.first_response_due_ts,
        ticket.resolution_due_ts,
        ticket.arrival_ts,
        ticket.ticket_id,
    )


def is_agent_feasible(ticket: TicketRecord, agent: AgentRecord) -> bool:
    """Check the hard v1 feasibility rules for one ticket-agent pair."""

    return (
        ticket.queue in agent.queue_permissions
        and ticket.language in agent.languages
        and ticket.priority in agent.priority_scope
    )


def run_greedy_baseline(
    tickets: list[TicketRecord], agents: list[AgentRecord]
) -> BaselineResult:
    """Replay the planning horizon and assign tickets with an online greedy rule.

    The dispatcher only considers tickets that have already arrived and agents
    that are idle in the current slot. If a ticket cannot start immediately, it
    stays in the open pool and may be reconsidered in a later slot or on a
    later business day.
    """

    if not tickets:
        raise ValueError("No tickets provided to greedy baseline.")
    if not agents:
        raise ValueError("No agents provided to greedy baseline.")

    released_tickets = sorted(
        tickets,
        key=lambda ticket: (ticket.release_ts, ticket.arrival_ts, ticket.ticket_id),
    )
    first_day = min(ticket.arrival_ts.date() for ticket in tickets)
    last_day = max(ticket.arrival_ts.date() for ticket in tickets)
    replay_days = business_days_inclusive(first_day, last_day)

    earliest_shift_start = min(agent.shift_start for agent in agents)
    latest_shift_end = max(agent.shift_end for agent in agents)
    horizon_start_ts = combine_date_and_time(replay_days[0], earliest_shift_start)
    final_horizon_end = combine_date_and_time(replay_days[-1], latest_shift_end)

    open_tickets: dict[str, TicketRecord] = {}
    schedule_by_ticket: dict[str, ScheduleEntry] = {}
    next_release_index = 0
    total_workload_minutes = {agent.agent_id: 0 for agent in agents}
    per_day_workload: dict[tuple[str, date], int] = defaultdict(int)

    for day in replay_days:
        # `busy_until` is reset per day because v1 assumes no overnight carry-over
        # for in-process work. A ticket is either finished today or still waiting.
        busy_until = {
            agent.agent_id: combine_date_and_time(day, agent.shift_start)
            for agent in agents
        }
        slot_starts = day_slot_starts(day, earliest_shift_start, latest_shift_end)

        for current_slot in slot_starts:
            # Move all tickets that have become available by this slot into the
            # open pool. They stay there until scheduled or until the replay ends.
            while (
                next_release_index < len(released_tickets)
                and released_tickets[next_release_index].release_ts <= current_slot
            ):
                ticket = released_tickets[next_release_index]
                open_tickets[ticket.ticket_id] = ticket
                next_release_index += 1

            if not open_tickets:
                continue

            idle_agents = [
                agent
                for agent in agents
                if busy_until[agent.agent_id] <= current_slot
                and current_slot >= combine_date_and_time(day, agent.shift_start)
                and current_slot < combine_date_and_time(day, agent.shift_end)
            ]
            if not idle_agents:
                continue

            # This is the online part of the heuristic: we only consider tickets
            # that can start right now, and we do not reserve future capacity.
            for ticket in sorted(open_tickets.values(), key=ticket_sort_key):
                eligible_agents: list[AgentRecord] = []
                processing_minutes = ticket.duration_slots * SLOT_MINUTES

                for agent in idle_agents:
                    if not is_agent_feasible(ticket, agent):
                        continue

                    if (
                        per_day_workload[(agent.agent_id, day)] + processing_minutes
                        > agent.capacity_min_per_day
                    ):
                        continue

                    completion_ts = current_slot + timedelta(minutes=processing_minutes)
                    agent_shift_end = combine_date_and_time(day, agent.shift_end)
                    if completion_ts > agent_shift_end:
                        continue

                    eligible_agents.append(agent)

                if not eligible_agents:
                    continue

                chosen_agent = min(
                    eligible_agents,
                    key=lambda agent: (
                        # Preserve the senior incident coordinator when another
                        # feasible agent can handle the work.
                        agent.scarce_resource,
                        per_day_workload[(agent.agent_id, day)],
                        agent.agent_id,
                    ),
                )

                completion_ts = current_slot + timedelta(minutes=processing_minutes)
                schedule_by_ticket[ticket.ticket_id] = ScheduleEntry(
                    ticket_id=ticket.ticket_id,
                    status="scheduled",
                    agent_id=chosen_agent.agent_id,
                    start_ts=current_slot,
                    completion_ts=completion_ts,
                    arrival_ts=ticket.arrival_ts,
                    queue=ticket.queue,
                    priority=ticket.priority,
                    language=ticket.language,
                    duration_slots=ticket.duration_slots,
                    first_response_due_ts=ticket.first_response_due_ts,
                    resolution_due_ts=ticket.resolution_due_ts,
                    first_response_tardiness_min=tardiness_minutes(
                        current_slot, ticket.first_response_due_ts
                    ),
                    resolution_tardiness_min=tardiness_minutes(
                        completion_ts, ticket.resolution_due_ts
                    ),
                )
                busy_until[chosen_agent.agent_id] = completion_ts
                per_day_workload[(chosen_agent.agent_id, day)] += processing_minutes
                total_workload_minutes[chosen_agent.agent_id] += processing_minutes
                del open_tickets[ticket.ticket_id]
                idle_agents = [
                    agent
                    for agent in idle_agents
                    if agent.agent_id != chosen_agent.agent_id
                ]
                if not idle_agents:
                    break

    # Include tickets that arrived in the final slots but never found a feasible
    # immediate start before the replay horizon ended.
    while (
        next_release_index < len(released_tickets)
        and released_tickets[next_release_index].release_ts <= final_horizon_end
    ):
        ticket = released_tickets[next_release_index]
        open_tickets[ticket.ticket_id] = ticket
        next_release_index += 1

    for ticket in open_tickets.values():
        schedule_by_ticket[ticket.ticket_id] = ScheduleEntry(
            ticket_id=ticket.ticket_id,
            status="backlog_end",
            agent_id="",
            start_ts=None,
            completion_ts=None,
            arrival_ts=ticket.arrival_ts,
            queue=ticket.queue,
            priority=ticket.priority,
            language=ticket.language,
            duration_slots=ticket.duration_slots,
            first_response_due_ts=ticket.first_response_due_ts,
            resolution_due_ts=ticket.resolution_due_ts,
            first_response_tardiness_min=None,
            resolution_tardiness_min=None,
        )

    ordered_schedule = [
        schedule_by_ticket[ticket.ticket_id]
        for ticket in sorted(tickets, key=lambda item: (item.arrival_ts, item.ticket_id))
    ]
    metrics = build_metrics(
        ordered_schedule,
        agents,
        len(replay_days),
        total_workload_minutes,
        horizon_start_ts,
        final_horizon_end,
    )
    return BaselineResult(schedule=ordered_schedule, metrics=metrics)


def build_metrics(
    schedule: list[ScheduleEntry],
    agents: list[AgentRecord],
    replay_day_count: int,
    workload_minutes_per_agent: dict[str, int],
    horizon_start_ts: datetime,
    final_horizon_end: datetime,
) -> dict[str, Any]:
    """Aggregate the schedule into overview metrics plus nested detail sections."""

    scheduled_metrics = empty_metric_section()
    backlog_metrics = empty_metric_section()

    for entry in schedule:
        effort_min = entry.duration_slots * SLOT_MINUTES
        if entry.status == "scheduled":
            accumulate_metric_section(
                scheduled_metrics,
                entry.priority,
                effort_min,
                entry.first_response_tardiness_min or 0.0,
                entry.resolution_tardiness_min or 0.0,
            )
            continue

        accumulate_metric_section(
            backlog_metrics,
            entry.priority,
            effort_min,
            tardiness_minutes(final_horizon_end, entry.first_response_due_ts),
            tardiness_minutes(final_horizon_end, entry.resolution_due_ts),
        )

    scheduled_metrics = finalize_metric_section(scheduled_metrics)
    backlog_metrics = finalize_metric_section(backlog_metrics)

    agent_utilization = {}
    for agent in agents:
        capacity_minutes = replay_day_count * agent.capacity_min_per_day
        workload_minutes = workload_minutes_per_agent[agent.agent_id]
        agent_utilization[agent.agent_id] = {
            "workload_minutes": workload_minutes,
            "capacity_minutes": capacity_minutes,
            "utilization": round(workload_minutes / capacity_minutes, 4),
        }

    return {
        "replay_business_days": replay_day_count,
        "slot_minutes": SLOT_MINUTES,
        "horizon_start_ts": format_timestamp(horizon_start_ts),
        "horizon_end_ts": format_timestamp(final_horizon_end),
        "total_tickets": len(schedule),
        "scheduled_tickets": scheduled_metrics["ticket_count"],
        "tickets_in_backlog": backlog_metrics["ticket_count"],
        "total_first_response_tardiness_min": round(
            scheduled_metrics["first_response_tardiness_min"]
            + backlog_metrics["first_response_tardiness_min"],
            2,
        ),
        "total_resolution_tardiness_min": round(
            scheduled_metrics["resolution_tardiness_min"]
            + backlog_metrics["resolution_tardiness_min"],
            2,
        ),
        "scheduled": scheduled_metrics,
        "backlog": backlog_metrics,
        "agent_utilization": agent_utilization,
    }


def write_baseline_outputs(
    result: BaselineResult,
    schedule_output_path: str | Path,
    metrics_output_path: str | Path,
) -> None:
    """Write the baseline schedule and metric summary to disk."""

    schedule_path = Path(schedule_output_path)
    metrics_path = Path(metrics_output_path)
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with schedule_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEDULE_FIELDNAMES)
        writer.writeheader()
        for entry in result.schedule:
            writer.writerow(entry.to_row())

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(result.metrics, handle, indent=2)
        handle.write("\n")


def run_greedy_baseline_from_csv(
    ticket_csv_path: str | Path, agent_csv_path: str | Path
) -> BaselineResult:
    """Convenience wrapper for the CLI script and tests."""

    tickets = load_tickets(ticket_csv_path)
    agents = load_agents(agent_csv_path)
    return run_greedy_baseline(tickets, agents)
