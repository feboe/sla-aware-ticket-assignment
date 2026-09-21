"""Shared business-time calendar for SLA generation, evaluation, and optimization.

The benchmark uses one support calendar for every SLA priority: Monday through
Friday, 08:00--16:00.  Time outside that window does not advance an SLA clock.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta

BUSINESS_WEEKDAYS = frozenset(range(5))
BUSINESS_START = time(8, 0)
BUSINESS_END = time(16, 0)
BUSINESS_DAY_MINUTES = (
    BUSINESS_END.hour * 60
    + BUSINESS_END.minute
    - BUSINESS_START.hour * 60
    - BUSINESS_START.minute
)


def business_day_bounds(day: date) -> tuple[datetime, datetime]:
    """Return the inclusive-start, exclusive-end business window for ``day``."""

    return datetime.combine(day, BUSINESS_START), datetime.combine(day, BUSINESS_END)


def is_business_timestamp(ts: datetime) -> bool:
    """Return whether ``ts`` lies inside or at the close of the business window."""

    day_start, day_end = business_day_bounds(ts.date())
    return ts.weekday() in BUSINESS_WEEKDAYS and day_start <= ts <= day_end


def next_business_day_start(ts: datetime) -> datetime:
    """Return the next business-day opening strictly after ``ts``'s date."""

    next_day = ts.date() + timedelta(days=1)
    while next_day.weekday() not in BUSINESS_WEEKDAYS:
        next_day += timedelta(days=1)
    return datetime.combine(next_day, BUSINESS_START)


def add_business_minutes(ts: datetime, minutes: int | float) -> datetime:
    """Advance ``ts`` by SLA minutes, pausing outside the business calendar."""

    if minutes < 0:
        raise ValueError("Business minutes to add must be non-negative.")

    current = ts
    remaining = float(minutes)
    while remaining > 0:
        day_start, day_end = business_day_bounds(current.date())
        if current.weekday() not in BUSINESS_WEEKDAYS:
            current = datetime.combine(current.date(), BUSINESS_START)
            while current.weekday() not in BUSINESS_WEEKDAYS:
                current += timedelta(days=1)
            continue
        if current < day_start:
            current = day_start
        elif current >= day_end:
            current = next_business_day_start(current)
            continue

        available = (day_end - current).total_seconds() / 60
        step = min(available, remaining)
        current += timedelta(minutes=step)
        remaining -= step
        if remaining > 0:
            current = next_business_day_start(current)

    return current


def business_minutes_between(start_ts: datetime, end_ts: datetime) -> float:
    """Return signed elapsed SLA minutes between two timestamps.

    Only overlap with weekday 08:00--16:00 windows is counted.  The function
    deliberately accepts timestamps outside that window so reports and tests
    have defined behaviour at calendar boundaries.
    """

    if end_ts == start_ts:
        return 0.0
    if end_ts < start_ts:
        return -business_minutes_between(end_ts, start_ts)

    total_minutes = 0.0
    current_day = start_ts.date()
    while current_day <= end_ts.date():
        if current_day.weekday() in BUSINESS_WEEKDAYS:
            day_start, day_end = business_day_bounds(current_day)
            overlap_start = max(start_ts, day_start)
            overlap_end = min(end_ts, day_end)
            if overlap_end > overlap_start:
                total_minutes += (overlap_end - overlap_start).total_seconds() / 60
        current_day += timedelta(days=1)
    return total_minutes


def business_slot_offset_floor(ts: datetime, origin: datetime, slot_minutes: int) -> int:
    """Convert a timestamp to a business-time slot offset using floor semantics."""

    if slot_minutes <= 0:
        raise ValueError("Slot duration must be positive.")
    return math.floor(business_minutes_between(origin, ts) / slot_minutes)
