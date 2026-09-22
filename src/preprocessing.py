"""Shared preprocessing utilities for ticket-assignment models and heuristics."""

import csv
import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
SLOT_MINUTES = 5


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
    """Supply-side agent capabilities used by dispatch and optimization logic."""

    agent_id: str
    queue_permissions: frozenset[str]
    languages: frozenset[str]
    priority_scope: frozenset[str]
    shift_start: time
    shift_end: time
    capacity_min_per_day: int
    scarce_resource: bool


def parse_timestamp(value: str) -> datetime:
    """Parse timestamps using the shared dataset format."""

    return datetime.strptime(value, TIMESTAMP_FORMAT)


def parse_pipe_set(value: str) -> frozenset[str]:
    """Parse pipe-delimited capability fields from the input CSVs."""

    return frozenset(part for part in value.split("|") if part)


def ceil_to_slot(ts: datetime) -> datetime:
    """Round a timestamp up to the next decision slot."""

    slot_floor = ts.replace(
        minute=(ts.minute // SLOT_MINUTES) * SLOT_MINUTES,
        second=0,
        microsecond=0,
    )
    if ts == slot_floor:
        return slot_floor
    return slot_floor + timedelta(minutes=SLOT_MINUTES)


def round_effort_to_slots(estimated_effort_min: int) -> int:
    """Convert effort minutes into a non-zero number of decision slots."""

    return max(1, math.ceil(estimated_effort_min / SLOT_MINUTES))


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
    """Enumerate slot starts across the active business window."""

    slot_starts: list[datetime] = []
    current = combine_date_and_time(day, earliest_shift_start)
    day_end = combine_date_and_time(day, latest_shift_end)
    while current < day_end:
        slot_starts.append(current)
        current += timedelta(minutes=SLOT_MINUTES)
    return slot_starts


def derive_replay_horizon(
    tickets: list[TicketRecord], agents: list[AgentRecord]
) -> tuple[list[date], datetime, datetime]:
    """Return replay business days plus the shared start and end timestamps.

    Policies and output validation must use the same definition of the replay
    horizon.  The caller is responsible for rejecting empty ticket or agent
    collections before invoking this helper.
    """

    if not tickets:
        raise ValueError("Cannot derive a replay horizon without tickets.")
    if not agents:
        raise ValueError("Cannot derive a replay horizon without agents.")

    first_day = min(ticket.arrival_ts.date() for ticket in tickets)
    last_day = max(ticket.arrival_ts.date() for ticket in tickets)
    replay_days = business_days_inclusive(first_day, last_day)
    if not replay_days:
        raise ValueError("Ticket horizon contains no business days.")

    earliest_shift_start = min(agent.shift_start for agent in agents)
    latest_shift_end = max(agent.shift_end for agent in agents)
    return (
        replay_days,
        combine_date_and_time(replay_days[0], earliest_shift_start),
        combine_date_and_time(replay_days[-1], latest_shift_end),
    )


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


def is_agent_feasible(ticket: TicketRecord, agent: AgentRecord) -> bool:
    """Check the hard v1 feasibility rules for one ticket-agent pair."""

    return (
        ticket.queue in agent.queue_permissions
        and ticket.language in agent.languages
        and ticket.priority in agent.priority_scope
    )
