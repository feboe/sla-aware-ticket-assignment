"""Preparation helpers for current-slot OR scheduler instances."""

import math
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
    parse_timestamp,
)


@dataclass(frozen=True)
class OrSchedulerInstance:
    """One current-slot optimization instance for the OR scheduler."""

    requested_decision_ts: datetime
    decision_ts: datetime
    horizon_end_ts: datetime
    horizon_slot_count: int
    tickets: tuple[TicketRecord, ...]
    agents: tuple[AgentRecord, ...]
    feasible_agent_ids: dict[str, tuple[str, ...]]
    remaining_capacity_minutes: dict[str, int]


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


def _can_start_ticket_now(
    ticket: TicketRecord,
    agent: AgentRecord,
    decision_ts: datetime,
    remaining_capacity_min: int,
    occupied_slots: set[datetime],
) -> bool:
    """Check whether one ticket-agent pair can legally start in the current slot."""

    if not is_agent_feasible(ticket, agent):
        return False

    processing_minutes = ticket.duration_slots * SLOT_MINUTES
    if remaining_capacity_min < processing_minutes:
        return False

    shift_start_ts = combine_date_and_time(decision_ts.date(), agent.shift_start)
    shift_end_ts = combine_date_and_time(decision_ts.date(), agent.shift_end)
    if decision_ts < max(ticket.release_ts, shift_start_ts):
        return False

    completion_ts = decision_ts + timedelta(minutes=processing_minutes)
    if completion_ts > shift_end_ts:
        return False

    return not _overlaps_committed_slots(
        decision_ts, ticket.duration_slots, occupied_slots
    )


def prepare_or_scheduler_instance(
    ticket_csv_path: str | Path | None,
    agent_csv_path: str | Path | None,
    decision_ts: str | datetime,
    *,
    tickets_override: list[TicketRecord] | None = None,
    agents_override: list[AgentRecord] | None = None,
    candidate_tickets: list[TicketRecord] | None = None,
    occupied_slots_by_agent: dict[tuple[str, date], set[datetime]] | None = None,
    used_capacity_minutes: dict[tuple[str, date], int] | None = None,
) -> OrSchedulerInstance:
    """Build one current-slot OR scheduler instance for a specific decision time."""

    requested_decision_ts = (
        decision_ts if isinstance(decision_ts, datetime) else parse_timestamp(decision_ts)
    )
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
        raise ValueError("No agents provided to the OR scheduler.")

    ordered_agents = tuple(sorted(agents_source, key=lambda agent: agent.agent_id))
    latest_shift_end = max(agent.shift_end for agent in ordered_agents)
    horizon_end_ts = combine_date_and_time(rounded_decision_ts.date(), latest_shift_end)
    horizon_slot_count = max(
        0,
        int(
            math.ceil(
                (horizon_end_ts - rounded_decision_ts).total_seconds() / 60 / SLOT_MINUTES
            )
        ),
    )
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

    occupied_slots_by_agent = occupied_slots_by_agent or {}
    used_capacity_minutes = used_capacity_minutes or {}
    remaining_capacity_minutes: dict[str, int] = {}
    feasible_agent_ids: dict[str, tuple[str, ...]] = {}

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
        feasible_agents = []
        for agent in ordered_agents:
            occupied_slots = occupied_slots_by_agent.get(
                (agent.agent_id, rounded_decision_ts.date()), set()
            )
            if _can_start_ticket_now(
                ticket,
                agent,
                rounded_decision_ts,
                remaining_capacity_minutes[agent.agent_id],
                occupied_slots,
            ):
                feasible_agents.append(agent.agent_id)
        feasible_agent_ids[ticket.ticket_id] = tuple(feasible_agents)

    return OrSchedulerInstance(
        requested_decision_ts=requested_decision_ts,
        decision_ts=rounded_decision_ts,
        horizon_end_ts=horizon_end_ts,
        horizon_slot_count=horizon_slot_count,
        tickets=active_tickets,
        agents=ordered_agents,
        feasible_agent_ids=feasible_agent_ids,
        remaining_capacity_minutes=remaining_capacity_minutes,
    )
