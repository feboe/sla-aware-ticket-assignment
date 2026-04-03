"""CP-SAT model construction and solving helpers for one-run OR instances."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ortools.sat.python import cp_model

from src.or_preparation import OrInstance, slot_offset_floor
from src.preprocessing import SLOT_MINUTES

FIRST_RESPONSE_WEIGHTS = {"P1": 1000, "P2": 200, "P3": 40, "P4": 10}
RESOLUTION_WEIGHTS = {"P1": 100, "P2": 20, "P3": 4, "P4": 1}
FAIRNESS_WEIGHT = 1


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
        model.Add(
            load[agent.agent_id]
            <= instance.remaining_capacity_minutes[agent.agent_id] // SLOT_MINUTES
        )

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
    instance: OrInstance, time_limit_sec: float = 30, num_workers: int = 8
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
