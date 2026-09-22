# Output Validation

Every final policy run validates its schedule and metrics before returning a
result. A valid result has exactly one `scheduled` or `backlog_end` row for every
input ticket. The validator checks preserved ticket data, agent eligibility,
release and slot timing, duration, shifts, same-day completion, overlap, daily
capacity, and business-minute tardiness. Violations are reported together in
deterministic order.

Metrics are independently rebuilt from the final schedule and compared to the
shared metric fields with a tolerance of `0.01`. Policy-specific diagnostics,
such as CP-SAT solver statistics, are allowed as additional fields. The layer
does not judge policy quality, objective values, tie-breaking, or skill tags;
skill tags remain metadata rather than a hard feasibility constraint.

## Stable violation codes

| Code | Meaning |
| --- | --- |
| `MISSING_TICKET`, `DUPLICATE_TICKET`, `UNKNOWN_TICKET` | Final rows do not map one-to-one to input tickets. |
| `INVALID_STATUS` | A final row is neither `scheduled` nor `backlog_end`. |
| `SCHEDULED_REQUIRED_FIELD`, `BACKLOG_FIELD_NOT_EMPTY` | Status-dependent fields are missing or unexpectedly populated. |
| `TICKET_FIELD_MISMATCH` | Persisted ticket master data differs from the input ticket. |
| `UNKNOWN_AGENT`, `INELIGIBLE_AGENT` | The assigned agent is absent or cannot serve the ticket. |
| `START_BEFORE_ARRIVAL`, `START_BEFORE_RELEASE`, `START_NOT_ON_SLOT` | The assigned start violates input timing or five-minute slotting. |
| `INVALID_DURATION`, `CROSS_DAY_COMPLETION`, `OUTSIDE_SHIFT` | Processing duration, same-day completion, or agent shift is invalid. |
| `AGENT_OVERLAP`, `CAPACITY_EXCEEDED` | An agent is double-booked or exceeds daily capacity. |
| `INVALID_TARDINESS` | Persisted ticket tardiness is not the business-calendar value. |
| `METRIC_MISMATCH` | A required shared metric is missing, non-finite, or differs by more than `0.01`. |
