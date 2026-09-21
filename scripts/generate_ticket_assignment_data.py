"""Generate the synthetic demand-side ticket dataset used in the project.

The generator keeps the data intentionally simple and explainable: tickets are
created during business hours only, SLAs are derived from priority, and the
output stays deterministic for a fixed random seed.
"""

import csv
import random
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

from src.preprocessing import TIMESTAMP_FORMAT
from src.business_calendar import (
    BUSINESS_DAY_MINUTES,
    BUSINESS_START,
    add_business_minutes,
    is_business_timestamp,
    next_business_day_start,
)

SEED = 42
OUTPUT_CSV = Path("data/tickets.csv")

# Kept as public generator constants for compatibility, but derived from the
# shared calendar so generated deadlines and validation cannot drift apart.
SHIFT_START_HOUR = BUSINESS_START.hour
SHIFT_DURATION_HOURS = BUSINESS_DAY_MINUTES // 60
N_BUSINESS_DAYS = 5
START_DATE = datetime(2026, 3, 2, 8, 0, 0)  # Monday

QUEUES = [
    "Account Access",
    "Billing",
    "Product Support",
    "Integrations/API",
]
QUEUE_WEIGHTS = [0.24, 0.16, 0.42, 0.18]

LANGUAGES = ["EN", "DE"]
LANGUAGE_WEIGHTS = [0.76, 0.24]

CHANNELS = ["portal", "email", "chat"]
CHANNEL_WEIGHTS = [0.54, 0.36, 0.10]

CUSTOMER_TIERS = ["SMB", "Mid-Market", "Enterprise"]
CUSTOMER_TIER_WEIGHTS = [0.52, 0.31, 0.17]

PRIORITIES = ["P1", "P2", "P3", "P4"]

FIRST_RESPONSE_SLA_MIN = {
    "P1": 15,
    "P2": 60,
    "P3": 240,
    "P4": 480,
}
RESOLUTION_SLA_BUSINESS_MIN = {
    "P1": 240,
    "P2": 480,
    "P3": 960,
    "P4": 2400,
}

QUEUE_SKILL = {
    "Account Access": "account_access",
    "Billing": "billing",
    "Product Support": "product_support",
    "Integrations/API": "integrations_api",
}

QUEUE_SUBJECTS: Dict[str, List[str]] = {
    "Account Access": [
        "User cannot access workspace after password reset",
        "MFA challenge not accepted for admin account",
        "SSO login loops back to sign-in page",
        "Role change not reflected after access approval",
        "New team member did not receive activation email",
        "Session expires immediately after successful login",
        "Permission update not reflected in project dashboard",
        "Account locked after repeated login attempts",
        "SCIM-provisioned user missing expected product access",
        "Group-based permissions not syncing from identity provider",
    ],
    "Billing": [
        "Invoice shows duplicate line items after plan change",
        "Upgrade request created prorated amount mismatch",
        "Payment method update failed during renewal",
        "Subscription cancellation not reflected in billing portal",
        "Purchase order reference missing from latest invoice",
        "Tax calculation differs from contract expectations",
        "Credit note not applied to outstanding balance",
        "Seat reduction request still billed at previous volume",
        "Billing contact did not receive monthly invoice email",
        "Currency setting changed but invoice still generated in USD",
    ],
    "Product Support": [
        "Dashboard filter not persisting for saved view",
        "Workflow automation did not trigger after status update",
        "Exported CSV misses custom fields in final report",
        "Bulk update operation stalled for large ticket batch",
        "Search results differ between UI and exported report",
        "Webhook retry settings not visible in admin interface",
        "Notification rule sent duplicate emails to requester",
        "Attachment preview fails for PDF uploaded from mobile app",
        "Custom field values not visible in agent workspace",
        "Kanban board shows outdated ticket counts after refresh",
    ],
    "Integrations/API": [
        "Webhook delivery returns intermittent 502 responses",
        "OAuth token refresh fails for production integration",
        "API rate limit reached earlier than documented threshold",
        "SSO assertion rejected after identity provider certificate update",
        "Webhook signature validation differs from documentation example",
        "GraphQL endpoint times out on nested customer query",
        "REST API pagination skips records after cursor handoff",
        "SCIM deprovisioning request returns 403 for service account",
        "CRM integration stopped syncing ticket comments overnight",
        "ERP connector creates duplicate records on retry",
    ],
}

QUEUE_PRIORITY_WEIGHTS = {
    "Account Access": [0.03, 0.16, 0.45, 0.36],
    "Billing": [0.02, 0.13, 0.46, 0.39],
    "Product Support": [0.05, 0.18, 0.50, 0.27],
    "Integrations/API": [0.09, 0.24, 0.46, 0.21],
}

COMPLEXITY_BY_QUEUE = {
    "Account Access": [0.46, 0.43, 0.11],  # simple, standard, complex
    "Billing": [0.39, 0.46, 0.15],
    "Product Support": [0.21, 0.55, 0.24],
    "Integrations/API": [0.08, 0.44, 0.48],
}

COMPLEXITIES = ["simple", "standard", "complex"]
EFFORT_RANGES = {
    ("Account Access", "simple"): (5, 12),
    ("Account Access", "standard"): (15, 30),
    ("Account Access", "complex"): (35, 75),
    ("Billing", "simple"): (6, 15),
    ("Billing", "standard"): (18, 35),
    ("Billing", "complex"): (40, 80),
    ("Product Support", "simple"): (10, 22),
    ("Product Support", "standard"): (25, 45),
    ("Product Support", "complex"): (50, 95),
    ("Integrations/API", "simple"): (15, 28),
    ("Integrations/API", "standard"): (30, 60),
    ("Integrations/API", "complex"): (60, 120),
}


@dataclass
class Ticket:
    """Flat ticket record written directly to the public CSV schema."""

    ticket_id: str
    arrival_ts: str
    queue: str
    priority: str
    language: str
    channel: str
    customer_tier: str
    is_vip: int
    is_reopened: int
    complexity: str
    estimated_effort_min: int
    first_response_sla_min: int
    first_response_due_ts: str
    resolution_sla_business_min: int
    resolution_due_ts: str
    required_skill_tags: str
    ticket_subject: str


def weighted_choice(rng: random.Random, values: List[str], weights: List[float]) -> str:
    """Sample one value from a weighted categorical distribution."""

    return rng.choices(values, weights=weights, k=1)[0]


def business_days(start: datetime, n: int) -> List[datetime]:
    """Return the first `n` weekday start timestamps from `start` onward."""

    days = []
    cur = start
    while len(days) < n:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def format_timestamp(ts: datetime) -> str:
    """Format timestamps using the shared CSV timestamp representation."""

    return ts.strftime(TIMESTAMP_FORMAT)


def parse_timestamp(value: str) -> datetime:
    """Parse timestamps from the shared CSV timestamp representation."""

    return datetime.strptime(value, TIMESTAMP_FORMAT)


def choose_priority(
    rng: random.Random,
    queue: str,
    customer_tier: str,
    is_vip: int,
    is_reopened: int,
    channel: str,
) -> str:
    """Sample a priority after applying simple business-driven adjustments."""

    weights = list(QUEUE_PRIORITY_WEIGHTS[queue])

    # Enterprise and VIP cases shift mass towards higher priority.
    if customer_tier == "Enterprise":
        weights[0] += 0.03
        weights[1] += 0.05
        weights[3] -= 0.05
    if is_vip:
        weights[0] += 0.05
        weights[1] += 0.04
        weights[3] -= 0.06
    if is_reopened:
        weights[1] += 0.04
        weights[2] += 0.02
        weights[3] -= 0.05
    if channel == "chat":
        weights[1] += 0.03
        weights[3] -= 0.02

    weights = [max(w, 0.01) for w in weights]
    total = sum(weights)
    norm = [w / total for w in weights]
    return weighted_choice(rng, PRIORITIES, norm)


def choose_complexity(
    rng: random.Random, queue: str, priority: str, is_reopened: int
) -> str:
    """Sample complexity with extra mass on harder reopened or urgent work."""

    weights = list(COMPLEXITY_BY_QUEUE[queue])
    if priority == "P1":
        weights[2] += 0.18
        weights[0] -= 0.08
    elif priority == "P2":
        weights[2] += 0.08
        weights[0] -= 0.03
    if is_reopened:
        weights[2] += 0.10
        weights[0] -= 0.04

    weights = [max(w, 0.02) for w in weights]
    total = sum(weights)
    norm = [w / total for w in weights]
    return weighted_choice(rng, COMPLEXITIES, norm)


def estimate_effort(
    rng: random.Random, queue: str, complexity: str, priority: str, is_vip: int
) -> int:
    """Draw handling effort in minutes from queue- and complexity-based ranges."""

    lo, hi = EFFORT_RANGES[(queue, complexity)]
    effort = rng.randint(lo, hi)
    if priority == "P1":
        effort = int(round(effort * 1.15))
    elif priority == "P2":
        effort = int(round(effort * 1.05))
    if is_vip and queue in {"Product Support", "Integrations/API"}:
        effort = int(round(effort * 1.05))
    return max(5, effort)


def make_required_skill_tags(
    queue: str,
    language: str,
    complexity: str,
    priority: str,
    customer_tier: str,
    is_vip: int,
) -> str:
    """Construct demand-side capability tags for later assignment logic."""

    tags = [QUEUE_SKILL[queue]]
    if language == "DE":
        tags.append("de_language")
    if customer_tier == "Enterprise" or is_vip:
        tags.append("enterprise_handling")
    if queue == "Integrations/API":
        tags.append("api_specialist")
    if priority == "P1":
        tags.append("incident_escalation")
    if queue == "Product Support" and complexity == "complex":
        tags.append("product_specialist")
    if queue == "Integrations/API" and complexity == "complex":
        tags.append("integration_specialist")
    return "|".join(tags)


def daily_ticket_count(rng: random.Random, idx: int) -> int:
    """Generate one day's ticket volume within the configured weekly range."""

    # Keeps totals in the requested range while still varying by day.
    base = rng.randint(80, 120)
    if idx in {1, 2}:  # Tue/Wed slightly busier in many support teams
        base += rng.randint(0, 6)
    return min(120, max(80, base))


def arrival_times_for_day(
    rng: random.Random, day_start: datetime, n: int
) -> List[datetime]:
    """Generate sorted within-day arrivals with a mild late-morning peak."""

    # Continuous arrivals during business hours with a mild concentration in late morning.
    seconds_total = SHIFT_DURATION_HOURS * 60 * 60
    arrivals = []
    for _ in range(n):
        # Mixture of two beta distributions to avoid overly uniform traffic.
        if rng.random() < 0.65:
            frac = rng.betavariate(2.3, 2.8)
        else:
            frac = rng.betavariate(1.4, 3.2)
        offset = int(frac * seconds_total)
        offset = min(seconds_total - 1, max(0, offset))
        arrivals.append(day_start + timedelta(seconds=offset))
    arrivals.sort()
    return arrivals


def generate_dataset(seed: int = SEED) -> List[Ticket]:
    """Generate the full synthetic ticket dataset for the configured horizon."""

    rng = random.Random(seed)
    days = business_days(START_DATE, N_BUSINESS_DAYS)
    rows: List[Ticket] = []
    ticket_counter = 1

    for day_idx, day_start in enumerate(days):
        n_tickets = daily_ticket_count(rng, day_idx)
        arrivals = arrival_times_for_day(rng, day_start, n_tickets)

        for arrival in arrivals:
            queue = weighted_choice(rng, QUEUES, QUEUE_WEIGHTS)
            language = weighted_choice(rng, LANGUAGES, LANGUAGE_WEIGHTS)
            channel = weighted_choice(rng, CHANNELS, CHANNEL_WEIGHTS)
            customer_tier = weighted_choice(rng, CUSTOMER_TIERS, CUSTOMER_TIER_WEIGHTS)
            is_vip = 1 if (customer_tier == "Enterprise" and rng.random() < 0.22) else 0
            is_reopened = 1 if rng.random() < 0.08 else 0

            priority = choose_priority(
                rng, queue, customer_tier, is_vip, is_reopened, channel
            )
            complexity = choose_complexity(rng, queue, priority, is_reopened)
            estimated_effort = estimate_effort(rng, queue, complexity, priority, is_vip)
            first_response_sla = FIRST_RESPONSE_SLA_MIN[priority]
            resolution_sla = RESOLUTION_SLA_BUSINESS_MIN[priority]
            first_response_due_ts = add_business_minutes(arrival, first_response_sla)
            resolution_due_ts = add_business_minutes(arrival, resolution_sla)
            required_skill_tags = make_required_skill_tags(
                queue, language, complexity, priority, customer_tier, is_vip
            )
            subject = rng.choice(QUEUE_SUBJECTS[queue])

            rows.append(
                Ticket(
                    ticket_id=f"TKT-{ticket_counter:05d}",
                    arrival_ts=arrival.strftime("%Y-%m-%d %H:%M:%S"),
                    queue=queue,
                    priority=priority,
                    language=language,
                    channel=channel,
                    customer_tier=customer_tier,
                    is_vip=is_vip,
                    is_reopened=is_reopened,
                    complexity=complexity,
                    estimated_effort_min=estimated_effort,
                    first_response_sla_min=first_response_sla,
                    first_response_due_ts=format_timestamp(first_response_due_ts),
                    resolution_sla_business_min=resolution_sla,
                    resolution_due_ts=format_timestamp(resolution_due_ts),
                    required_skill_tags=required_skill_tags,
                    ticket_subject=subject,
                )
            )
            ticket_counter += 1

    return rows


def validate_dataset(rows: List[Ticket]) -> None:
    """Enforce the public data contract before writing the CSV."""

    if not rows:
        raise ValueError("Generated dataset is empty.")

    seen_ticket_ids = set()
    previous_arrival: datetime | None = None

    for row in rows:
        for field in fields(Ticket):
            value = getattr(row, field.name)
            if isinstance(value, str) and not value:
                raise ValueError(f"Field {field.name} is empty for {row.ticket_id}.")
            if value is None:
                raise ValueError(f"Field {field.name} is None for {row.ticket_id}.")

        if row.ticket_id in seen_ticket_ids:
            raise ValueError(f"Duplicate ticket_id detected: {row.ticket_id}")
        seen_ticket_ids.add(row.ticket_id)

        arrival = parse_timestamp(row.arrival_ts)
        first_response_due = parse_timestamp(row.first_response_due_ts)
        resolution_due = parse_timestamp(row.resolution_due_ts)

        if previous_arrival and arrival < previous_arrival:
            raise ValueError("arrival_ts values are not monotonically ordered.")
        previous_arrival = arrival

        if not (arrival <= first_response_due <= resolution_due):
            raise ValueError(
                "Deadline order violated for "
                f"{row.ticket_id}: {row.arrival_ts}, {row.first_response_due_ts}, "
                f"{row.resolution_due_ts}"
            )

        for ts_name, ts_value in (
            ("arrival_ts", arrival),
            ("first_response_due_ts", first_response_due),
            ("resolution_due_ts", resolution_due),
        ):
            if not is_business_timestamp(ts_value):
                raise ValueError(
                    f"{ts_name} falls outside business calendar for {row.ticket_id}: "
                    f"{format_timestamp(ts_value)}"
                )


def write_csv(rows: List[Ticket], output_path: Path = OUTPUT_CSV) -> None:
    """Write validated ticket rows to the configured output CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[field.name for field in fields(Ticket)],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


if __name__ == "__main__":
    rows = generate_dataset(SEED)
    validate_dataset(rows)
    write_csv(rows, OUTPUT_CSV)
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
