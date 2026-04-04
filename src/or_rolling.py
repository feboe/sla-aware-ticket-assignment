"""Rolling-horizon replay wrapper around the one-run CP-SAT ticket-assignment model."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from src.evaluation import SCHEDULE_FIELDNAMES, ScheduleEntry
from src.greedy_baseline import build_metrics
from src.or_preparation import prepare_or_instance
from src.or_reporting import extract_or_schedule
from src.or_solver import solve_cp_sat_instance
from src.preprocessing import (
    AgentRecord,
    SLOT_MINUTES,
    TicketRecord,
    business_days_inclusive,
    combine_date_and_time,
    day_slot_starts,
    load_agents,
    load_tickets,
)

DEFAULT_ROLLING_TIME_LIMIT_SEC = 4.0


@dataclass(frozen=True)
class RollingOrResult:
    """Replay-style schedule and metrics for the rolling OR benchmark."""

    schedule: list[ScheduleEntry]
    metrics: dict[str, Any]


def _agent_can_start_now(
    agent: AgentRecord,
    day: date,
    current_slot: datetime,
    occupied_slots_by_agent: dict[tuple[str, date], set[datetime]],
    used_capacity_minutes: dict[tuple[str, date], int],
) -> bool:
    """Check whether an agent has capacity to begin a ticket in the current slot."""

    shift_start_ts = combine_date_and_time(day, agent.shift_start)
    shift_end_ts = combine_date_and_time(day, agent.shift_end)
    if current_slot < shift_start_ts or current_slot >= shift_end_ts:
        return False
    if current_slot in occupied_slots_by_agent[(agent.agent_id, day)]:
        return False
    return used_capacity_minutes[(agent.agent_id, day)] < agent.capacity_min_per_day


def run_rolling_or_model(
    tickets: list[TicketRecord],
    agents: list[AgentRecord],
    time_limit_sec: float = DEFAULT_ROLLING_TIME_LIMIT_SEC,
    num_workers: int = 8,
) -> RollingOrResult:
    """Replay the full dataset and re-solve a one-run OR model every 15 minutes."""

    if not tickets:
        raise ValueError("No tickets provided to the rolling OR benchmark.")
    if not agents:
        raise ValueError("No agents provided to the rolling OR benchmark.")

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
    occupied_slots_by_agent: dict[tuple[str, date], set[datetime]] = defaultdict(set)
    solver_status_counts: Counter[str] = Counter()
    solve_call_count = 0
    total_solve_time_sec = 0.0
    slot_delta = timedelta(minutes=SLOT_MINUTES)

    for day in replay_days:
        slot_starts = day_slot_starts(day, earliest_shift_start, latest_shift_end)

        for current_slot in slot_starts:
            while (
                next_release_index < len(released_tickets)
                and released_tickets[next_release_index].release_ts <= current_slot
            ):
                ticket = released_tickets[next_release_index]
                if ticket.ticket_id not in schedule_by_ticket:
                    open_tickets[ticket.ticket_id] = ticket
                next_release_index += 1

            if not open_tickets:
                continue

            if not any(
                _agent_can_start_now(
                    agent,
                    day,
                    current_slot,
                    occupied_slots_by_agent,
                    per_day_workload,
                )
                for agent in agents
            ):
                continue

            instance = prepare_or_instance(
                None,
                None,
                current_slot,
                tickets_override=tickets,
                agents_override=agents,
                candidate_tickets=list(open_tickets.values()),
                occupied_slots_by_agent=occupied_slots_by_agent,
                used_capacity_minutes=per_day_workload,
                current_slot_only_starts=True,
            )
            if not instance.tickets:
                continue

            artifacts = solve_cp_sat_instance(
                instance, time_limit_sec=time_limit_sec, num_workers=num_workers
            )
            solve_call_count += 1
            total_solve_time_sec += artifacts.solve_time_sec
            solver_status_counts[artifacts.status_name] += 1

            for entry in extract_or_schedule(artifacts):
                if entry.status != "scheduled" or entry.start_ts != current_slot:
                    continue

                schedule_by_ticket[entry.ticket_id] = entry
                processing_minutes = entry.duration_slots * SLOT_MINUTES
                probe = entry.start_ts
                while probe < entry.completion_ts:
                    occupied_slots_by_agent[(entry.agent_id, day)].add(probe)
                    probe += slot_delta
                per_day_workload[(entry.agent_id, day)] += processing_minutes
                total_workload_minutes[entry.agent_id] += processing_minutes
                open_tickets.pop(entry.ticket_id, None)

    while (
        next_release_index < len(released_tickets)
        and released_tickets[next_release_index].release_ts <= final_horizon_end
    ):
        ticket = released_tickets[next_release_index]
        if ticket.ticket_id not in schedule_by_ticket:
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
    metrics["solve_call_count"] = solve_call_count
    metrics["avg_solve_time_sec"] = (
        round(total_solve_time_sec / solve_call_count, 4) if solve_call_count else 0.0
    )
    metrics["solver_status_counts"] = dict(sorted(solver_status_counts.items()))
    return RollingOrResult(schedule=ordered_schedule, metrics=metrics)


def write_rolling_or_outputs(
    result: RollingOrResult,
    schedule_output_path: str | Path,
    metrics_output_path: str | Path,
) -> None:
    """Write the rolling OR replay outputs to disk."""

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


def run_rolling_or_model_from_csv(
    ticket_csv_path: str | Path,
    agent_csv_path: str | Path,
    time_limit_sec: float = DEFAULT_ROLLING_TIME_LIMIT_SEC,
    num_workers: int = 8,
) -> RollingOrResult:
    """Convenience wrapper for the rolling OR CLI and tests."""

    tickets = load_tickets(ticket_csv_path)
    agents = load_agents(agent_csv_path)
    return run_rolling_or_model(
        tickets,
        agents,
        time_limit_sec=time_limit_sec,
        num_workers=num_workers,
    )
