"""Shared schedule serialization and metric helpers for ticket-assignment outputs."""

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.business_calendar import business_minutes_between
from src.preprocessing import AgentRecord, SLOT_MINUTES, TIMESTAMP_FORMAT

SCHEDULE_FIELDNAMES = [
    "ticket_id",
    "status",
    "agent_id",
    "start_ts",
    "completion_ts",
    "arrival_ts",
    "queue",
    "priority",
    "language",
    "duration_slots",
    "first_response_due_ts",
    "resolution_due_ts",
    "first_response_tardiness_min",
    "resolution_tardiness_min",
]
PRIORITY_BUCKETS = ("P1", "P2", "P3", "P4")


def format_timestamp(value: datetime | None) -> str:
    """Serialize datetimes back into the shared dataset format."""

    if value is None:
        return ""
    return value.strftime(TIMESTAMP_FORMAT)


def format_metric_value(value: float | None) -> str:
    """Format numeric metrics for CSV output while preserving empty backlog rows."""

    if value is None:
        return ""
    return f"{value:.2f}"


def tardiness_minutes(actual_ts: datetime, due_ts: datetime) -> float:
    """Return positive SLA lateness in business minutes.

    The SLA clock pauses outside the shared Monday--Friday, 08:00--16:00
    business calendar.  This applies equally to scheduled work and backlog
    evaluated at the final replay horizon.
    """

    return round(max(0.0, business_minutes_between(due_ts, actual_ts)), 2)


def empty_priority_tardiness_metrics() -> dict[str, float | int]:
    """Build one empty metric bucket for a single priority class."""

    return {
        "ticket_count": 0,
        "effort_min": 0,
        "first_response_tardiness_min": 0.0,
        "resolution_tardiness_min": 0.0,
        "overdue_first_response_count": 0,
        "overdue_resolution_count": 0,
    }


def empty_metric_section() -> dict[str, Any]:
    """Build one empty metric section shared by scheduled and backlog views."""

    return {
        "ticket_count": 0,
        "effort_min": 0,
        "first_response_tardiness_min": 0.0,
        "resolution_tardiness_min": 0.0,
        "overdue_first_response_count": 0,
        "overdue_resolution_count": 0,
        "by_priority": {
            priority: empty_priority_tardiness_metrics() for priority in PRIORITY_BUCKETS
        },
    }


def accumulate_metric_section(
    section: dict[str, Any],
    priority: str,
    effort_min: int,
    first_response_tardiness: float,
    resolution_tardiness: float,
) -> None:
    """Accumulate one ticket's contribution into a metric section."""

    priority_bucket = section["by_priority"][priority]

    section["ticket_count"] += 1
    section["effort_min"] += effort_min
    section["first_response_tardiness_min"] += first_response_tardiness
    section["resolution_tardiness_min"] += resolution_tardiness
    priority_bucket["ticket_count"] += 1
    priority_bucket["effort_min"] += effort_min
    priority_bucket["first_response_tardiness_min"] += first_response_tardiness
    priority_bucket["resolution_tardiness_min"] += resolution_tardiness

    if first_response_tardiness > 0:
        section["overdue_first_response_count"] += 1
        priority_bucket["overdue_first_response_count"] += 1
    if resolution_tardiness > 0:
        section["overdue_resolution_count"] += 1
        priority_bucket["overdue_resolution_count"] += 1


def finalize_metric_section(section: dict[str, Any]) -> dict[str, Any]:
    """Round float values while keeping the shared metric schema stable."""

    return {
        "ticket_count": section["ticket_count"],
        "effort_min": section["effort_min"],
        "first_response_tardiness_min": round(section["first_response_tardiness_min"], 2),
        "resolution_tardiness_min": round(section["resolution_tardiness_min"], 2),
        "overdue_first_response_count": section["overdue_first_response_count"],
        "overdue_resolution_count": section["overdue_resolution_count"],
        "by_priority": {
            priority: {
                "ticket_count": values["ticket_count"],
                "effort_min": values["effort_min"],
                "first_response_tardiness_min": round(
                    values["first_response_tardiness_min"], 2
                ),
                "resolution_tardiness_min": round(values["resolution_tardiness_min"], 2),
                "overdue_first_response_count": values["overdue_first_response_count"],
                "overdue_resolution_count": values["overdue_resolution_count"],
            }
            for priority, values in section["by_priority"].items()
        },
    }


def compute_overall_agent_utilization(
    agents: list[AgentRecord],
    replay_day_count: int,
    workload_minutes_per_agent: dict[str, int],
) -> dict[str, float | int]:
    """Aggregate total workload and capacity across all agents."""

    workload_minutes = sum(workload_minutes_per_agent.values())
    capacity_minutes = sum(
        replay_day_count * agent.capacity_min_per_day for agent in agents
    )
    utilization = 0.0
    if capacity_minutes > 0:
        utilization = round(workload_minutes / capacity_minutes, 4)

    return {
        "workload_minutes": workload_minutes,
        "workload_hours": round(workload_minutes / 60.0, 2),
        "capacity_minutes": capacity_minutes,
        "capacity_hours": round(capacity_minutes / 60.0, 2),
        "utilization": utilization,
    }


def compute_agent_solved_priority_counts(
    schedule: list["ScheduleEntry"], agents: list[AgentRecord]
) -> dict[str, dict[str, int]]:
    """Count scheduled tickets per agent and priority using the final replay schedule."""

    counts = {
        agent.agent_id: {
            "p1_tickets_solved": 0,
            "p2_tickets_solved": 0,
            "p3_tickets_solved": 0,
            "p4_tickets_solved": 0,
        }
        for agent in agents
    }
    priority_key_by_label = {
        "P1": "p1_tickets_solved",
        "P2": "p2_tickets_solved",
        "P3": "p3_tickets_solved",
        "P4": "p4_tickets_solved",
    }

    for entry in schedule:
        if entry.status != "scheduled":
            continue
        counts[entry.agent_id][priority_key_by_label[entry.priority]] += 1

    return counts


@dataclass(frozen=True)
class ScheduleEntry:
    """Final schedule row for one ticket, either scheduled or left in backlog."""

    ticket_id: str
    status: str
    agent_id: str
    start_ts: datetime | None
    completion_ts: datetime | None
    arrival_ts: datetime
    queue: str
    priority: str
    language: str
    duration_slots: int
    first_response_due_ts: datetime
    resolution_due_ts: datetime
    first_response_tardiness_min: float | None
    resolution_tardiness_min: float | None

    def to_row(self) -> dict[str, str]:
        """Convert the schedule entry into a CSV-ready string dictionary."""

        return {
            "ticket_id": self.ticket_id,
            "status": self.status,
            "agent_id": self.agent_id,
            "start_ts": format_timestamp(self.start_ts),
            "completion_ts": format_timestamp(self.completion_ts),
            "arrival_ts": format_timestamp(self.arrival_ts),
            "queue": self.queue,
            "priority": self.priority,
            "language": self.language,
            "duration_slots": str(self.duration_slots),
            "first_response_due_ts": format_timestamp(self.first_response_due_ts),
            "resolution_due_ts": format_timestamp(self.resolution_due_ts),
            "first_response_tardiness_min": format_metric_value(
                self.first_response_tardiness_min
            ),
            "resolution_tardiness_min": format_metric_value(
                self.resolution_tardiness_min
            ),
        }


def compute_schedule_metrics(
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

    solved_priority_counts = compute_agent_solved_priority_counts(schedule, agents)
    agent_utilization: dict[str, dict[str, float | int]] = {}
    for agent in agents:
        capacity_minutes = replay_day_count * agent.capacity_min_per_day
        workload_minutes = workload_minutes_per_agent[agent.agent_id]
        agent_utilization[agent.agent_id] = {
            "workload_minutes": workload_minutes,
            "capacity_minutes": capacity_minutes,
            "utilization": round(workload_minutes / capacity_minutes, 4),
            **solved_priority_counts[agent.agent_id],
        }
    overall_agent_utilization = compute_overall_agent_utilization(
        agents, replay_day_count, workload_minutes_per_agent
    )

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
        "overall_agent_utilization": overall_agent_utilization,
    }


def write_schedule_outputs(
    schedule: list[ScheduleEntry],
    metrics: dict[str, Any],
    schedule_output_path: str | Path,
    metrics_output_path: str | Path,
) -> None:
    """Write the shared schedule CSV and metrics JSON output files."""

    schedule_path = Path(schedule_output_path)
    metrics_path = Path(metrics_output_path)
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with schedule_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SCHEDULE_FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        for entry in schedule:
            writer.writerow(entry.to_row())

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
