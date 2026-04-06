"""Look-ahead greedy benchmark for the synthetic ticket-assignment problem.

This heuristic keeps the same urgency ordering and output schema as the simple
online baseline, but it can reserve the earliest feasible future slot later in
the same day instead of only assigning work that can start immediately.
"""

from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

from src.evaluation import ScheduleEntry, tardiness_minutes
from src.greedy_baseline import (
    BaselineResult,
    compute_metrics,
    ticket_sort_key,
)
from src.preprocessing import (
    AgentRecord,
    SLOT_MINUTES,
    TicketRecord,
    business_days_inclusive,
    combine_date_and_time,
    day_slot_starts,
    is_agent_feasible,
    load_agents,
    load_tickets,
)


def find_earliest_feasible_assignment(
    ticket: TicketRecord,
    agents: list[AgentRecord],
    day: date,
    current_slot: datetime,
    slot_starts: list[datetime],
    reserved_slots: dict[tuple[str, date], set[datetime]],
    per_day_workload: dict[tuple[str, date], int],
) -> tuple[AgentRecord, datetime] | None:
    """Return the earliest legal same-day placement for one open ticket."""

    processing_minutes = ticket.duration_slots * SLOT_MINUTES
    slot_delta = timedelta(minutes=SLOT_MINUTES)
    best_candidate: tuple[datetime, int, int, str, AgentRecord] | None = None

    for agent in sorted(agents, key=lambda item: item.agent_id):
        if not is_agent_feasible(ticket, agent):
            continue

        if (
            per_day_workload[(agent.agent_id, day)] + processing_minutes
            > agent.capacity_min_per_day
        ):
            continue

        shift_start_ts = combine_date_and_time(day, agent.shift_start)
        shift_end_ts = combine_date_and_time(day, agent.shift_end)
        earliest_start = max(current_slot, ticket.release_ts, shift_start_ts)
        occupied_slots = reserved_slots[(agent.agent_id, day)]

        for slot_start in slot_starts:
            if slot_start < earliest_start:
                continue

            completion_ts = slot_start + timedelta(minutes=processing_minutes)
            if completion_ts > shift_end_ts:
                break

            probe = slot_start
            while probe < completion_ts and probe not in occupied_slots:
                probe += slot_delta

            if probe < completion_ts:
                continue

            candidate = (
                slot_start,
                int(agent.scarce_resource),
                per_day_workload[(agent.agent_id, day)],
                agent.agent_id,
                agent,
            )
            if best_candidate is None or candidate[:4] < best_candidate[:4]:
                best_candidate = candidate
            break

    if best_candidate is None:
        return None
    return best_candidate[4], best_candidate[0]


def run_lookahead_greedy(
    tickets: list[TicketRecord], agents: list[AgentRecord]
) -> BaselineResult:
    """Replay the horizon with same-day earliest-fit look-ahead reservations."""

    if not tickets:
        raise ValueError("No tickets provided to look-ahead greedy baseline.")
    if not agents:
        raise ValueError("No agents provided to look-ahead greedy baseline.")

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
    reserved_slots: dict[tuple[str, date], set[datetime]] = defaultdict(set)
    slot_delta = timedelta(minutes=SLOT_MINUTES)

    for day in replay_days:
        slot_starts = day_slot_starts(day, earliest_shift_start, latest_shift_end)

        for current_slot in slot_starts:
            while (
                next_release_index < len(released_tickets)
                and released_tickets[next_release_index].release_ts <= current_slot
            ):
                ticket = released_tickets[next_release_index]
                open_tickets[ticket.ticket_id] = ticket
                next_release_index += 1

            if not open_tickets:
                continue

            for ticket in sorted(open_tickets.values(), key=ticket_sort_key):
                assignment = find_earliest_feasible_assignment(
                    ticket,
                    agents,
                    day,
                    current_slot,
                    slot_starts,
                    reserved_slots,
                    per_day_workload,
                )
                if assignment is None:
                    continue

                chosen_agent, start_ts = assignment
                processing_minutes = ticket.duration_slots * SLOT_MINUTES
                completion_ts = start_ts + timedelta(minutes=processing_minutes)

                probe = start_ts
                while probe < completion_ts:
                    reserved_slots[(chosen_agent.agent_id, day)].add(probe)
                    probe += slot_delta

                schedule_by_ticket[ticket.ticket_id] = ScheduleEntry(
                    ticket_id=ticket.ticket_id,
                    status="scheduled",
                    agent_id=chosen_agent.agent_id,
                    start_ts=start_ts,
                    completion_ts=completion_ts,
                    arrival_ts=ticket.arrival_ts,
                    queue=ticket.queue,
                    priority=ticket.priority,
                    language=ticket.language,
                    duration_slots=ticket.duration_slots,
                    first_response_due_ts=ticket.first_response_due_ts,
                    resolution_due_ts=ticket.resolution_due_ts,
                    first_response_tardiness_min=tardiness_minutes(
                        start_ts, ticket.first_response_due_ts
                    ),
                    resolution_tardiness_min=tardiness_minutes(
                        completion_ts, ticket.resolution_due_ts
                    ),
                )
                per_day_workload[(chosen_agent.agent_id, day)] += processing_minutes
                total_workload_minutes[chosen_agent.agent_id] += processing_minutes
                del open_tickets[ticket.ticket_id]

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
    metrics = compute_metrics(
        ordered_schedule,
        agents,
        len(replay_days),
        total_workload_minutes,
        horizon_start_ts,
        final_horizon_end,
    )
    return BaselineResult(schedule=ordered_schedule, metrics=metrics)


def run_lookahead_greedy_from_csv(
    ticket_csv_path: str | Path, agent_csv_path: str | Path
) -> BaselineResult:
    """Convenience wrapper for the look-ahead benchmark CLI and tests."""

    tickets = load_tickets(ticket_csv_path)
    agents = load_agents(agent_csv_path)
    return run_lookahead_greedy(tickets, agents)
