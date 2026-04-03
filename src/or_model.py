"""Minimal one-run CP-SAT model for the synthetic ticket-assignment problem.

This module keeps the first exact optimization implementation deliberately small:
it solves a single decision timestamp over the remaining slots of that workday.
The model stays close to the written formulation and reuses the shared
preprocessing and evaluation helpers.
"""

from __future__ import annotations

import csv
import json
import math
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

FIRST_RESPONSE_WEIGHTS = {"P1": 1000, "P2": 200, "P3": 40, "P4": 10}
RESOLUTION_WEIGHTS = {"P1": 100, "P2": 20, "P3": 4, "P4": 1}
FAIRNESS_WEIGHT = 1


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


@dataclass(frozen=True)
class OrModelVariables:
    """Decision and auxiliary variables used by the CP-SAT model."""

    x: dict[tuple[str, str, int], cp_model.IntVar]
    backlog: dict[str, cp_model.IntVar]
    first_response_tardiness: dict[str, cp_model.IntVar]
    resolution_tardiness: dict[str, cp_model.IntVar]
    load: dict[str, cp_model.IntVar]
    max_load: cp_model.IntVar
    min_load: cp_model.IntVar


@dataclass(frozen=True)
class OrSolveArtifacts:
    """Solver output plus the variables needed for schedule extraction."""

    instance: OrInstance
    variables: OrModelVariables
    solver: cp_model.CpSolver
    status_code: int
    status_name: str
    objective_value: float | None
    solve_time_sec: float


@dataclass(frozen=True)
class OrResult:
    """Container for the one-run OR schedule and aggregate metrics."""

    schedule: list[ScheduleEntry]
    metrics: dict[str, Any]


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
    return math.floor(delta_minutes / SLOT_MINUTES)


def prepare_or_instance(
    ticket_csv_path: str | Path,
    agent_csv_path: str | Path,
    decision_ts: str | datetime,
) -> OrInstance:
    """Load CSV data and derive the one-run CP-SAT instance for a given timestamp."""

    requested_decision_ts = parse_decision_timestamp(decision_ts)
    rounded_decision_ts = ceil_to_slot(requested_decision_ts)
    tickets = load_tickets(ticket_csv_path)
    agents = load_agents(agent_csv_path)

    if not agents:
        raise ValueError("No agents provided to the OR model.")

    latest_shift_end = max(agent.shift_end for agent in agents)
    horizon_end_ts = combine_date_and_time(rounded_decision_ts.date(), latest_shift_end)
    slot_starts = build_slot_starts(rounded_decision_ts, horizon_end_ts)

    active_tickets = tuple(
        sorted(
            (ticket for ticket in tickets if ticket.release_ts <= rounded_decision_ts),
            key=lambda ticket: (ticket.arrival_ts, ticket.ticket_id),
        )
    )
    ordered_agents = tuple(sorted(agents, key=lambda agent: agent.agent_id))

    feasible_agent_ids: dict[str, tuple[str, ...]] = {}
    allowed_start_indices: dict[tuple[str, str], tuple[int, ...]] = {}
    remaining_capacity_minutes: dict[str, int] = {}

    for agent in ordered_agents:
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
        remaining_capacity_minutes[agent.agent_id] = min(
            agent.capacity_min_per_day, remaining_shift_minutes
        )

    for ticket in active_tickets:
        feasible_agents = tuple(
            agent.agent_id for agent in ordered_agents if is_agent_feasible(ticket, agent)
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

            allowed_indices: list[int] = []
            for start_index, slot_start in enumerate(slot_starts):
                completion_ts = slot_start + timedelta(
                    minutes=ticket.duration_slots * SLOT_MINUTES
                )
                if slot_start < earliest_start:
                    continue
                if completion_ts > shift_end_ts:
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


def build_cp_sat_model(instance: OrInstance) -> tuple[cp_model.CpModel, OrModelVariables]:
    """Build the minimal time-indexed CP-SAT model for one decision timestamp."""

    model = cp_model.CpModel()
    x: dict[tuple[str, str, int], cp_model.IntVar] = {}
    backlog: dict[str, cp_model.IntVar] = {}
    first_response_tardiness: dict[str, cp_model.IntVar] = {}
    resolution_tardiness: dict[str, cp_model.IntVar] = {}
    load: dict[str, cp_model.IntVar] = {}

    total_duration_slots = sum(ticket.duration_slots for ticket in instance.tickets)
    horizon_slot_count = len(instance.slot_starts)
    max_duration_slots = max(
        (ticket.duration_slots for ticket in instance.tickets), default=0
    )
    earliest_due_ts = min(
        (
            min(ticket.first_response_due_ts, ticket.resolution_due_ts)
            for ticket in instance.tickets
        ),
        default=instance.horizon_end_ts,
    )
    max_tardiness_slots = (
        max(
            0,
            math.ceil(
                (instance.horizon_end_ts - earliest_due_ts).total_seconds()
                / 60
                / SLOT_MINUTES
            ),
        )
        + max_duration_slots
    )

    for ticket in instance.tickets:
        backlog[ticket.ticket_id] = model.NewBoolVar(f"backlog_{ticket.ticket_id}")

        for agent in instance.agents:
            for start_index in instance.allowed_start_indices[
                (ticket.ticket_id, agent.agent_id)
            ]:
                x[(ticket.ticket_id, agent.agent_id, start_index)] = model.NewBoolVar(
                    f"start_{ticket.ticket_id}_{agent.agent_id}_{start_index}"
                )

        first_response_tardiness[ticket.ticket_id] = model.NewIntVar(
            0, max_tardiness_slots, f"u_fr_{ticket.ticket_id}"
        )
        resolution_tardiness[ticket.ticket_id] = model.NewIntVar(
            0, max_tardiness_slots, f"u_res_{ticket.ticket_id}"
        )

    for agent in instance.agents:
        load[agent.agent_id] = model.NewIntVar(
            0, total_duration_slots, f"load_{agent.agent_id}"
        )

    max_load = model.NewIntVar(0, total_duration_slots, "max_load")
    min_load = model.NewIntVar(0, total_duration_slots, "min_load")

    for ticket in instance.tickets:
        ticket_start_vars = [
            x[(ticket.ticket_id, agent.agent_id, start_index)]
            for agent in instance.agents
            for start_index in instance.allowed_start_indices[
                (ticket.ticket_id, agent.agent_id)
            ]
        ]
        model.Add(sum(ticket_start_vars) + backlog[ticket.ticket_id] == 1)

        first_response_due_offset = slot_offset_floor(
            ticket.first_response_due_ts, instance.horizon_start_ts
        )
        resolution_due_offset = slot_offset_floor(
            ticket.resolution_due_ts, instance.horizon_start_ts
        )
        start_expression = sum(
            start_index * x[(ticket.ticket_id, agent.agent_id, start_index)]
            for agent in instance.agents
            for start_index in instance.allowed_start_indices[
                (ticket.ticket_id, agent.agent_id)
            ]
        )
        completion_expression = sum(
            (start_index + ticket.duration_slots)
            * x[(ticket.ticket_id, agent.agent_id, start_index)]
            for agent in instance.agents
            for start_index in instance.allowed_start_indices[
                (ticket.ticket_id, agent.agent_id)
            ]
        )

        # Backlogged tickets incur end-of-horizon lateness so the model cannot
        # escape the objective by deferring everything.
        model.Add(
            first_response_tardiness[ticket.ticket_id]
            >= start_expression
            + horizon_slot_count * backlog[ticket.ticket_id]
            - first_response_due_offset
        )
        model.Add(
            resolution_tardiness[ticket.ticket_id]
            >= completion_expression
            + horizon_slot_count * backlog[ticket.ticket_id]
            - resolution_due_offset
        )

    for agent in instance.agents:
        for occupied_slot in range(horizon_slot_count):
            overlapping_starts = [
                x[(ticket.ticket_id, agent.agent_id, start_index)]
                for ticket in instance.tickets
                for start_index in instance.allowed_start_indices[
                    (ticket.ticket_id, agent.agent_id)
                ]
                if start_index <= occupied_slot < start_index + ticket.duration_slots
            ]
            if overlapping_starts:
                model.Add(sum(overlapping_starts) <= 1)

        agent_starts = [
            ticket.duration_slots * x[(ticket.ticket_id, agent.agent_id, start_index)]
            for ticket in instance.tickets
            for start_index in instance.allowed_start_indices[
                (ticket.ticket_id, agent.agent_id)
            ]
        ]
        model.Add(load[agent.agent_id] == sum(agent_starts))
        model.Add(load[agent.agent_id] <= max_load)
        model.Add(load[agent.agent_id] >= min_load)

    objective_terms: list[Any] = []
    for ticket in instance.tickets:
        objective_terms.append(
            FIRST_RESPONSE_WEIGHTS[ticket.priority]
            * first_response_tardiness[ticket.ticket_id]
        )
        objective_terms.append(
            RESOLUTION_WEIGHTS[ticket.priority] * resolution_tardiness[ticket.ticket_id]
        )
    objective_terms.append(FAIRNESS_WEIGHT * (max_load - min_load))
    model.Minimize(sum(objective_terms))

    return model, OrModelVariables(
        x=x,
        backlog=backlog,
        first_response_tardiness=first_response_tardiness,
        resolution_tardiness=resolution_tardiness,
        load=load,
        max_load=max_load,
        min_load=min_load,
    )


def solve_cp_sat_instance(
    instance: OrInstance, time_limit_sec: int = 30, num_workers: int = 8
) -> OrSolveArtifacts:
    """Solve one OR instance and capture solver metadata for extraction."""

    model, variables = build_cp_sat_model(instance)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = num_workers
    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code)
    objective_value = None
    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        objective_value = solver.ObjectiveValue()

    return OrSolveArtifacts(
        instance=instance,
        variables=variables,
        solver=solver,
        status_code=status_code,
        status_name=status_name,
        objective_value=objective_value,
        solve_time_sec=solver.WallTime(),
    )


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


def run_or_model_from_csv(
    ticket_csv_path: str | Path,
    agent_csv_path: str | Path,
    decision_ts: str | datetime,
    time_limit_sec: int = 30,
    num_workers: int = 8,
) -> OrResult:
    """Run the one-run OR model from CSV inputs and return schedule plus metrics."""

    instance = prepare_or_instance(ticket_csv_path, agent_csv_path, decision_ts)
    artifacts = solve_cp_sat_instance(
        instance, time_limit_sec=time_limit_sec, num_workers=num_workers
    )
    schedule = extract_or_schedule(artifacts)
    metrics = build_or_metrics(
        schedule,
        instance,
        solver_status=artifacts.status_name,
        objective_value=artifacts.objective_value,
        solve_time_sec=artifacts.solve_time_sec,
    )
    return OrResult(schedule=schedule, metrics=metrics)
