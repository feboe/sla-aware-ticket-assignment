"""Compatibility facade for the one-run CP-SAT ticket-assignment model."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.or_preparation import (
    OrInstance,
    build_slot_starts,
    parse_decision_timestamp,
    prepare_or_instance,
    slot_offset_floor,
)
from src.or_reporting import (
    OrResult,
    build_or_metrics,
    extract_or_schedule,
    write_or_outputs,
)
from src.or_solver import (
    FAIRNESS_WEIGHT,
    FIRST_RESPONSE_WEIGHTS,
    RESOLUTION_WEIGHTS,
    OrModelVariables,
    OrSolveArtifacts,
    build_cp_sat_model,
    solve_cp_sat_instance,
)


def run_or_model_from_csv(
    ticket_csv_path: str | Path,
    agent_csv_path: str | Path,
    decision_ts: str | datetime,
    time_limit_sec: float = 30,
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


__all__ = [
    "FAIRNESS_WEIGHT",
    "FIRST_RESPONSE_WEIGHTS",
    "RESOLUTION_WEIGHTS",
    "OrInstance",
    "OrModelVariables",
    "OrResult",
    "OrSolveArtifacts",
    "build_cp_sat_model",
    "build_or_metrics",
    "build_slot_starts",
    "extract_or_schedule",
    "parse_decision_timestamp",
    "prepare_or_instance",
    "run_or_model_from_csv",
    "slot_offset_floor",
    "solve_cp_sat_instance",
    "write_or_outputs",
]
