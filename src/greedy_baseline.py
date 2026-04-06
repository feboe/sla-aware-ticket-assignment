"""Greedy baseline for the synthetic ticket-assignment problem.

This module keeps the online dispatch policy only. Shared preprocessing, schedule
serialization, and metric helpers live in dedicated modules so the baseline and
the OR model can both depend on the same source of truth.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from src.evaluation import (
    ScheduleEntry,
    compute_schedule_metrics,
    tardiness_minutes,
    write_schedule_outputs,
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

PRIORITY_RANK = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}


@dataclass(frozen=True)
class BaselineResult:
    """Container for the ticket-level schedule and aggregate baseline metrics."""

    schedule: list[ScheduleEntry]
    metrics: dict[str, Any]


def ticket_sort_key(ticket: TicketRecord) -> tuple[Any, ...]:
    """Return the deterministic urgency order used by the greedy policy."""

    return (
        PRIORITY_RANK[ticket.priority],
        ticket.first_response_due_ts,
        ticket.resolution_due_ts,
        ticket.arrival_ts,
        ticket.ticket_id,
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
    metrics = compute_metrics(
        ordered_schedule,
        agents,
        len(replay_days),
        total_workload_minutes,
        horizon_start_ts,
        final_horizon_end,
    )
    return BaselineResult(schedule=ordered_schedule, metrics=metrics)


def compute_metrics(
    schedule: list[ScheduleEntry],
    agents: list[AgentRecord],
    replay_day_count: int,
    workload_minutes_per_agent: dict[str, int],
    horizon_start_ts: datetime,
    final_horizon_end: datetime,
) -> dict[str, Any]:
    """Aggregate the schedule into overview metrics plus nested detail sections."""

    return compute_schedule_metrics(
        schedule,
        agents,
        replay_day_count,
        workload_minutes_per_agent,
        horizon_start_ts,
        final_horizon_end,
    )


def write_baseline_outputs(
    result: BaselineResult,
    schedule_output_path: str | Path,
    metrics_output_path: str | Path,
) -> None:
    """Write the baseline schedule and metric summary to disk."""

    write_schedule_outputs(
        result.schedule, result.metrics, schedule_output_path, metrics_output_path
    )


def run_greedy_baseline_from_csv(
    ticket_csv_path: str | Path, agent_csv_path: str | Path
) -> BaselineResult:
    """Convenience wrapper for the CLI script and tests."""

    tickets = load_tickets(ticket_csv_path)
    agents = load_agents(agent_csv_path)
    return run_greedy_baseline(tickets, agents)


__all__ = [
    "BaselineResult",
    "ticket_sort_key",
    "run_greedy_baseline",
    "compute_metrics",
    "write_baseline_outputs",
    "run_greedy_baseline_from_csv",
]
