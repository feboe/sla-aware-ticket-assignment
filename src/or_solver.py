"""CP-SAT model construction and solving helpers for the OR scheduler."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ortools.sat.python import cp_model

from src.or_preparation import OrSchedulerInstance
from src.preprocessing import SLOT_MINUTES

FIRST_RESPONSE_WEIGHTS = {"P1": 1000, "P2": 200, "P3": 40, "P4": 10}
RESOLUTION_WEIGHTS = {"P1": 100, "P2": 20, "P3": 4, "P4": 1}
BACKLOG_WEIGHT = 0.5
SCARCE_AGENT_WEIGHT = 0.5


@dataclass(frozen=True)
class OrSchedulerVariables:
    """Decision variables used by the current-slot OR scheduler model."""

    x: dict[tuple[str, str], cp_model.IntVar]
    backlog: dict[str, cp_model.IntVar]
    first_response_tardiness: dict[str, cp_model.IntVar]
    resolution_tardiness: dict[str, cp_model.IntVar]


@dataclass(frozen=True)
class OrSchedulerSolveArtifacts:
    """Solver output plus the variables needed for schedule extraction."""

    instance: OrSchedulerInstance
    variables: OrSchedulerVariables
    solver: cp_model.CpSolver
    status_code: int
    status_name: str
    objective_value: float | None
    solve_time_sec: float


def _slot_offset_floor(ts, origin) -> int:
    """Convert a timestamp to a slot offset relative to ``origin`` using floor semantics."""

    delta_minutes = (ts - origin).total_seconds() / 60.0
    return int(delta_minutes // SLOT_MINUTES)


def create_or_scheduler_model(
    instance: OrSchedulerInstance,
) -> tuple[cp_model.CpModel, OrSchedulerVariables]:
    """Create the current-slot CP-SAT model for one scheduler decision."""

    model = cp_model.CpModel()
    agent_by_id = {agent.agent_id: agent for agent in instance.agents}
    x: dict[tuple[str, str], cp_model.IntVar] = {}
    backlog: dict[str, cp_model.IntVar] = {}
    first_response_tardiness: dict[str, cp_model.IntVar] = {}
    resolution_tardiness: dict[str, cp_model.IntVar] = {}

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
        first_response_tardiness[ticket.ticket_id] = model.NewIntVar(
            0, max_tardiness_slots, f"u_fr_{ticket.ticket_id}"
        )
        resolution_tardiness[ticket.ticket_id] = model.NewIntVar(
            0, max_tardiness_slots, f"u_res_{ticket.ticket_id}"
        )
        for agent_id in instance.feasible_agent_ids[ticket.ticket_id]:
            x[(ticket.ticket_id, agent_id)] = model.NewBoolVar(
                f"x_{ticket.ticket_id}_{agent_id}"
            )

    for ticket in instance.tickets:
        ticket_assignments = [
            x[(ticket.ticket_id, agent_id)]
            for agent_id in instance.feasible_agent_ids[ticket.ticket_id]
        ]
        model.Add(sum(ticket_assignments) + backlog[ticket.ticket_id] == 1)

        first_response_due_offset = _slot_offset_floor(
            ticket.first_response_due_ts, instance.decision_ts
        )
        resolution_due_offset = _slot_offset_floor(
            ticket.resolution_due_ts, instance.decision_ts
        )
        completion_expression = sum(
            ticket.duration_slots * x[(ticket.ticket_id, agent_id)]
            for agent_id in instance.feasible_agent_ids[ticket.ticket_id]
        )
        model.Add(
            first_response_tardiness[ticket.ticket_id]
            >= instance.horizon_slot_count * backlog[ticket.ticket_id]
            - first_response_due_offset
        )
        model.Add(
            resolution_tardiness[ticket.ticket_id]
            >= completion_expression
            + instance.horizon_slot_count * backlog[ticket.ticket_id]
            - resolution_due_offset
        )

    for agent in instance.agents:
        agent_assignments = [
            x[(ticket.ticket_id, agent.agent_id)]
            for ticket in instance.tickets
            if agent.agent_id in instance.feasible_agent_ids[ticket.ticket_id]
        ]
        if agent_assignments:
            model.Add(sum(agent_assignments) <= 1)

    objective_terms: list[Any] = []
    for ticket in instance.tickets:
        feasible_agent_ids = instance.feasible_agent_ids[ticket.ticket_id]
        objective_terms.append(
            FIRST_RESPONSE_WEIGHTS[ticket.priority]
            * first_response_tardiness[ticket.ticket_id]
        )
        objective_terms.append(
            RESOLUTION_WEIGHTS[ticket.priority] * resolution_tardiness[ticket.ticket_id]
        )
        objective_terms.append(BACKLOG_WEIGHT * backlog[ticket.ticket_id])
        has_non_scarce_alternative = any(
            not agent_by_id[agent_id].scarce_resource for agent_id in feasible_agent_ids
        )
        if has_non_scarce_alternative:
            for agent_id in feasible_agent_ids:
                if agent_by_id[agent_id].scarce_resource:
                    objective_terms.append(
                        SCARCE_AGENT_WEIGHT * x[(ticket.ticket_id, agent_id)]
                    )

    model.Minimize(sum(objective_terms))
    return model, OrSchedulerVariables(
        x=x,
        backlog=backlog,
        first_response_tardiness=first_response_tardiness,
        resolution_tardiness=resolution_tardiness,
    )


def solve_or_scheduler_instance(
    instance: OrSchedulerInstance,
    time_limit_sec: float = 30,
    num_workers: int = 8,
) -> OrSchedulerSolveArtifacts:
    """Solve one current-slot OR scheduler instance."""

    model, variables = create_or_scheduler_model(instance)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = num_workers
    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code)
    objective_value = None
    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        objective_value = solver.ObjectiveValue()

    return OrSchedulerSolveArtifacts(
        instance=instance,
        variables=variables,
        solver=solver,
        status_code=status_code,
        status_name=status_name,
        objective_value=objective_value,
        solve_time_sec=solver.WallTime(),
    )
