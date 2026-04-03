"""Reporting helpers for one-run CP-SAT ticket-assignment results."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import timedelta
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
from src.or_preparation import OrInstance
from src.or_solver import OrSolveArtifacts
from src.preprocessing import SLOT_MINUTES


@dataclass(frozen=True)
class OrResult:
    """Container for the one-run OR schedule and aggregate metrics."""

    schedule: list[ScheduleEntry]
    metrics: dict[str, Any]


def extract_or_schedule(artifacts: OrSolveArtifacts) -> list[ScheduleEntry]:
    """Translate the selected CP-SAT decisions back into schedule rows."""

    instance = artifacts.instance
    schedule: list[ScheduleEntry] = []
    has_solution = artifacts.status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    for ticket in instance.tickets:
        chosen_assignment: tuple[str, int] | None = None
        if has_solution:
            for agent in instance.agents:
                for start_index in instance.allowed_start_indices[
                    (ticket.ticket_id, agent.agent_id)
                ]:
                    if artifacts.solver.Value(
                        artifacts.variables.x[
                            (ticket.ticket_id, agent.agent_id, start_index)
                        ]
                    ):
                        chosen_assignment = (agent.agent_id, start_index)
                        break
                if chosen_assignment is not None:
                    break

        if chosen_assignment is None:
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

        agent_id, start_index = chosen_assignment
        start_ts = instance.slot_starts[start_index]
        completion_ts = start_ts + timedelta(minutes=ticket.duration_slots * SLOT_MINUTES)
        schedule.append(
            ScheduleEntry(
                ticket_id=ticket.ticket_id,
                status="scheduled",
                agent_id=agent_id,
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


def build_or_metrics(
    schedule: list[ScheduleEntry],
    instance: OrInstance,
    solver_status: str,
    objective_value: float | None,
    solve_time_sec: float,
) -> dict[str, Any]:
    """Aggregate the one-run schedule into overview and nested detail sections."""

    scheduled_metrics = empty_metric_section()
    backlog_metrics = empty_metric_section()
    workload_minutes_per_agent = {agent.agent_id: 0 for agent in instance.agents}

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
            workload_minutes_per_agent[entry.agent_id] += effort_min
            continue

        accumulate_metric_section(
            backlog_metrics,
            entry.priority,
            effort_min,
            tardiness_minutes(instance.horizon_end_ts, entry.first_response_due_ts),
            tardiness_minutes(instance.horizon_end_ts, entry.resolution_due_ts),
        )

    scheduled_metrics = finalize_metric_section(scheduled_metrics)
    backlog_metrics = finalize_metric_section(backlog_metrics)

    agent_utilization: dict[str, dict[str, float | int]] = {}
    for agent in instance.agents:
        capacity_minutes = instance.remaining_capacity_minutes[agent.agent_id]
        workload_minutes = workload_minutes_per_agent[agent.agent_id]
        utilization = 0.0
        if capacity_minutes > 0:
            utilization = round(workload_minutes / capacity_minutes, 4)
        agent_utilization[agent.agent_id] = {
            "workload_minutes": workload_minutes,
            "capacity_minutes": capacity_minutes,
            "utilization": utilization,
        }

    return {
        "slot_minutes": SLOT_MINUTES,
        "decision_ts": format_timestamp(instance.requested_decision_ts),
        "horizon_start_ts": format_timestamp(instance.horizon_start_ts),
        "horizon_end_ts": format_timestamp(instance.horizon_end_ts),
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
        "solver_status": solver_status,
        "objective_value": None if objective_value is None else round(objective_value, 2),
        "solve_time_sec": round(solve_time_sec, 4),
        "scheduled": scheduled_metrics,
        "backlog": backlog_metrics,
        "agent_utilization": agent_utilization,
    }


def write_or_outputs(
    result: OrResult,
    schedule_output_path: str | Path,
    metrics_output_path: str | Path,
) -> None:
    """Write the one-run OR schedule and metrics to disk."""

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
