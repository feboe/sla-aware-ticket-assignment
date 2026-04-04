"""CP-SAT model construction and solving helpers for the OR scheduler."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ortools.sat.python import cp_model

from src.or_preparation import OrSchedulerInstance
from src.preprocessing import SLOT_MINUTES

FIRST_RESPONSE_WEIGHTS = {"P1": 1000, "P2": 200, "P3": 40, "P4": 10}
RESOLUTION_WEIGHTS = {"P1": 100, "P2": 20, "P3": 4, "P4": 1}


@dataclass(frozen=True)
class OrSchedulerVariables:
    """Decision variables used by the current-slot OR scheduler model."""

    assign: dict[tuple[str, str], cp_model.IntVar]
    backlog: dict[str, cp_model.IntVar]


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


def _tardiness_slots(actual_ts: datetime, due_ts: datetime) -> int:
    """Return positive tardiness in scheduler slots."""

    delta_minutes = max(0.0, (actual_ts - due_ts).total_seconds() / 60.0)
    return int(math.ceil(delta_minutes / SLOT_MINUTES))


def _weighted_tardiness_cost(ticket, start_ts: datetime) -> int:
    """Return the weighted tardiness cost of starting the ticket at ``start_ts``."""

    completion_ts = start_ts + timedelta(minutes=ticket.duration_slots * SLOT_MINUTES)
    return FIRST_RESPONSE_WEIGHTS[ticket.priority] * _tardiness_slots(
        start_ts, ticket.first_response_due_ts
    ) + RESOLUTION_WEIGHTS[ticket.priority] * _tardiness_slots(
        completion_ts, ticket.resolution_due_ts
    )


def build_or_scheduler_model(
    instance: OrSchedulerInstance,
) -> tuple[cp_model.CpModel, OrSchedulerVariables]:
    """Build the current-slot CP-SAT model for one scheduler decision."""

    model = cp_model.CpModel()
    assign: dict[tuple[str, str], cp_model.IntVar] = {}
    backlog: dict[str, cp_model.IntVar] = {}

    for ticket in instance.tickets:
        backlog[ticket.ticket_id] = model.NewBoolVar(f"backlog_{ticket.ticket_id}")
        for agent_id in instance.feasible_agent_ids[ticket.ticket_id]:
            assign[(ticket.ticket_id, agent_id)] = model.NewBoolVar(
                f"assign_{ticket.ticket_id}_{agent_id}"
            )

    for ticket in instance.tickets:
        ticket_assignments = [
            assign[(ticket.ticket_id, agent_id)]
            for agent_id in instance.feasible_agent_ids[ticket.ticket_id]
        ]
        model.Add(sum(ticket_assignments) + backlog[ticket.ticket_id] == 1)

    for agent in instance.agents:
        agent_assignments = [
            assign[(ticket.ticket_id, agent.agent_id)]
            for ticket in instance.tickets
            if agent.agent_id in instance.feasible_agent_ids[ticket.ticket_id]
        ]
        if agent_assignments:
            model.Add(sum(agent_assignments) <= 1)

    objective_terms: list[Any] = []
    for ticket in instance.tickets:
        scheduled_cost = _weighted_tardiness_cost(ticket, instance.decision_ts)
        backlog_cost = _weighted_tardiness_cost(ticket, instance.next_decision_ts)
        for agent_id in instance.feasible_agent_ids[ticket.ticket_id]:
            objective_terms.append(scheduled_cost * assign[(ticket.ticket_id, agent_id)])
        objective_terms.append(backlog_cost * backlog[ticket.ticket_id])

    model.Minimize(sum(objective_terms))
    return model, OrSchedulerVariables(assign=assign, backlog=backlog)


def solve_or_scheduler_instance(
    instance: OrSchedulerInstance,
    time_limit_sec: float = 30,
    num_workers: int = 8,
) -> OrSchedulerSolveArtifacts:
    """Solve one current-slot OR scheduler instance."""

    model, variables = build_or_scheduler_model(instance)
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
