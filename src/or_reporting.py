"""Reporting helpers for current-slot OR scheduler results."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ortools.sat.python import cp_model

from src.evaluation import (
    ScheduleEntry,
    compute_schedule_metrics,
    tardiness_minutes,
    write_schedule_outputs,
)
from src.or_solver import OrSchedulerSolveArtifacts
from src.preprocessing import AgentRecord, SLOT_MINUTES


@dataclass(frozen=True)
class OrSchedulerResult:
    """Container for the scheduler replay outputs."""

    schedule: list[ScheduleEntry]
    metrics: dict[str, Any]


def extract_or_scheduler_schedule(
    artifacts: OrSchedulerSolveArtifacts,
) -> list[ScheduleEntry]:
    """Translate CP-SAT decisions back into current-slot schedule rows."""

    instance = artifacts.instance
    schedule: list[ScheduleEntry] = []
    has_solution = artifacts.status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    for ticket in instance.tickets:
        chosen_agent_id: str | None = None
        if has_solution:
            for agent_id in instance.feasible_agent_ids[ticket.ticket_id]:
                if artifacts.solver.Value(
                    artifacts.variables.x[(ticket.ticket_id, agent_id)]
                ):
                    chosen_agent_id = agent_id
                    break

        if chosen_agent_id is None:
            schedule.append(
                ScheduleEntry(
                    ticket_id=ticket.ticket_id,
                    status="backlog_current_run",
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
            )
            continue

        start_ts = instance.decision_ts
        completion_ts = start_ts + timedelta(minutes=ticket.duration_slots * SLOT_MINUTES)
        schedule.append(
            ScheduleEntry(
                ticket_id=ticket.ticket_id,
                status="scheduled",
                agent_id=chosen_agent_id,
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
        )

    return schedule


def compute_or_scheduler_metrics(
    schedule: list[ScheduleEntry],
    agents: list[AgentRecord],
    replay_day_count: int,
    workload_minutes_per_agent: dict[str, int],
    horizon_start_ts: datetime,
    final_horizon_end: datetime,
) -> dict[str, Any]:
    """Aggregate the replay schedule into the shared output metric schema."""

    return compute_schedule_metrics(
        schedule,
        agents,
        replay_day_count,
        workload_minutes_per_agent,
        horizon_start_ts,
        final_horizon_end,
    )


def write_or_scheduler_outputs(
    result: OrSchedulerResult,
    schedule_output_path: str | Path,
    metrics_output_path: str | Path,
) -> None:
    """Write the OR scheduler replay outputs to disk."""

    write_schedule_outputs(
        result.schedule, result.metrics, schedule_output_path, metrics_output_path
    )
