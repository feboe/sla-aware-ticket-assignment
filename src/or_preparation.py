"""Preparation helpers for one-run CP-SAT ticket-assignment instances."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from src.preprocessing import (
    AgentRecord,
    SLOT_MINUTES,
    TicketRecord,
    ceil_to_slot,
    combine_date_and_time,
    is_agent_feasible,
    load_agents,
    load_tickets,
)


@dataclass(frozen=True)
class OrInstance:
    """One optimization instance for a single decision timestamp."""

    requested_decision_ts: datetime
    horizon_start_ts: datetime
    horizon_end_ts: datetime
    slot_starts: tuple[datetime, ...]
    tickets: tuple[TicketRecord, ...]
    agents: tuple[AgentRecord, ...]
    feasible_agent_ids: dict[str, tuple[str, ...]]
    allowed_start_indices: dict[tuple[str, str], tuple[int, ...]]
    remaining_capacity_minutes: dict[str, int]


def parse_decision_timestamp(value: str | datetime) -> datetime:
    """Accept either a parsed timestamp or the shared timestamp string format."""

    if isinstance(value, datetime):
        return value
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def build_slot_starts(
    horizon_start_ts: datetime, horizon_end_ts: datetime
) -> tuple[datetime, ...]:
    """Enumerate all 15-minute slot starts within the one-run planning horizon."""

    slot_starts: list[datetime] = []
    current = horizon_start_ts
    while current < horizon_end_ts:
        slot_starts.append(current)
        current += timedelta(minutes=SLOT_MINUTES)
    return tuple(slot_starts)


def slot_offset_floor(ts: datetime, origin: datetime) -> int:
    """Convert a timestamp to a slot offset relative to origin using floor semantics."""

    delta_minutes = (ts - origin).total_seconds() / 60.0
    return int(delta_minutes // SLOT_MINUTES)


def _overlaps_committed_slots(
    slot_start: datetime,
    duration_slots: int,
    occupied_slots: set[datetime],
) -> bool:
    """Check whether a candidate start would overlap already committed work."""

    probe = slot_start
    slot_delta = timedelta(minutes=SLOT_MINUTES)
    for _ in range(duration_slots):
        if probe in occupied_slots:
            return True
        probe += slot_delta
    return False


def prepare_or_instance(
    ticket_csv_path: str | Path | None,
    agent_csv_path: str | Path | None,
    decision_ts: str | datetime,
    *,
    tickets_override: list[TicketRecord] | None = None,
    agents_override: list[AgentRecord] | None = None,
    candidate_tickets: list[TicketRecord] | None = None,
    occupied_slots_by_agent: dict[tuple[str, date], set[datetime]] | None = None,
    used_capacity_minutes: dict[tuple[str, date], int] | None = None,
) -> OrInstance:
    """Load or reuse parsed data and derive a one-run CP-SAT instance."""

    requested_decision_ts = parse_decision_timestamp(decision_ts)
    rounded_decision_ts = ceil_to_slot(requested_decision_ts)

    tickets_source = (
        list(tickets_override)
        if tickets_override is not None
        else load_tickets(ticket_csv_path)
    )
    agents_source = (
        list(agents_override)
        if agents_override is not None
        else load_agents(agent_csv_path)
    )

    if not agents_source:
        raise ValueError("No agents provided to the OR model.")

    latest_shift_end = max(agent.shift_end for agent in agents_source)
    horizon_end_ts = combine_date_and_time(rounded_decision_ts.date(), latest_shift_end)
    slot_starts = build_slot_starts(rounded_decision_ts, horizon_end_ts)
    candidate_source = (
        candidate_tickets if candidate_tickets is not None else tickets_source
    )
    active_tickets = tuple(
        sorted(
            (
                ticket
                for ticket in candidate_source
                if ticket.release_ts <= rounded_decision_ts
            ),
            key=lambda ticket: (ticket.arrival_ts, ticket.ticket_id),
        )
    )
    ordered_agents = tuple(sorted(agents_source, key=lambda agent: agent.agent_id))

    occupied_slots_by_agent = occupied_slots_by_agent or {}
    used_capacity_minutes = used_capacity_minutes or {}
    feasible_agent_ids: dict[str, tuple[str, ...]] = {}
    allowed_start_indices: dict[tuple[str, str], tuple[int, ...]] = {}
    remaining_capacity_minutes: dict[str, int] = {}

    for agent in ordered_agents:
        day_key = (agent.agent_id, rounded_decision_ts.date())
        shift_start_ts = combine_date_and_time(
            rounded_decision_ts.date(), agent.shift_start
        )
        shift_end_ts = combine_date_and_time(rounded_decision_ts.date(), agent.shift_end)
        remaining_shift_minutes = max(
            0,
            int(
                (shift_end_ts - max(rounded_decision_ts, shift_start_ts)).total_seconds()
                / 60
            ),
        )
        used_minutes = used_capacity_minutes.get(day_key, 0)
        remaining_capacity_minutes[agent.agent_id] = max(
            0,
            min(agent.capacity_min_per_day - used_minutes, remaining_shift_minutes),
        )

    for ticket in active_tickets:
        feasible_agents = tuple(
            agent.agent_id
            for agent in ordered_agents
            if is_agent_feasible(ticket, agent)
            and remaining_capacity_minutes[agent.agent_id]
            >= ticket.duration_slots * SLOT_MINUTES
        )
        feasible_agent_ids[ticket.ticket_id] = feasible_agents

        for agent in ordered_agents:
            key = (ticket.ticket_id, agent.agent_id)
            if agent.agent_id not in feasible_agents:
                allowed_start_indices[key] = ()
                continue

            shift_start_ts = combine_date_and_time(
                rounded_decision_ts.date(), agent.shift_start
            )
            shift_end_ts = combine_date_and_time(
                rounded_decision_ts.date(), agent.shift_end
            )
            earliest_start = max(rounded_decision_ts, ticket.release_ts, shift_start_ts)
            occupied_slots = occupied_slots_by_agent.get(
                (agent.agent_id, rounded_decision_ts.date()), set()
            )

            allowed_indices: list[int] = []
            for start_index, slot_start in enumerate(slot_starts):
                completion_ts = slot_start + timedelta(
                    minutes=ticket.duration_slots * SLOT_MINUTES
                )
                if slot_start < earliest_start:
                    continue
                if completion_ts > shift_end_ts:
                    continue
                if _overlaps_committed_slots(
                    slot_start, ticket.duration_slots, occupied_slots
                ):
                    continue
                allowed_indices.append(start_index)

            allowed_start_indices[key] = tuple(allowed_indices)

    return OrInstance(
        requested_decision_ts=requested_decision_ts,
        horizon_start_ts=rounded_decision_ts,
        horizon_end_ts=horizon_end_ts,
        slot_starts=slot_starts,
        tickets=active_tickets,
        agents=ordered_agents,
        feasible_agent_ids=feasible_agent_ids,
        allowed_start_indices=allowed_start_indices,
        remaining_capacity_minutes=remaining_capacity_minutes,
    )
