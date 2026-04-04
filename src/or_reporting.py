"""Reporting helpers for current-slot OR scheduler results."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ortools.sat.python import cp_model

from src.evaluation import (
    SCHEDULE_FIELDNAMES,
    ScheduleEntry,
    accumulate_metric_section,
    empty_metric_section,
    finalize_metric_section,
    format_timestamp,
    tardiness_minutes,
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

    agent_utilization: dict[str, dict[str, float | int]] = {}
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


def write_or_scheduler_outputs(
    result: OrSchedulerResult,
    schedule_output_path: str | Path,
    metrics_output_path: str | Path,
) -> None:
    """Write the OR scheduler replay outputs to disk."""

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
