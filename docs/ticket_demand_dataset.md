# Ticket Demand Dataset

## Use Case

This dataset represents incoming support-ticket demand for a service-operations
environment. It is synthetic by design and intended for experimentation,
benchmarking, and later optimization modeling.

This ticket file does not include agent supply, staffing, or assignment
outcomes. Each row is a demand-side record only. Agent supply is defined
separately in `data/agents.csv`.

## File

- `data/tickets.csv`

## Business Story

- A B2B software support team handles tickets across four queues:
  `Account Access`, `Billing`, `Product Support`, and `Integrations/API`.
- Tickets arrive during weekday business hours only.
- Priority influences both first-response and resolution expectations.
- Ticket characteristics such as queue, language, customer tier, complexity, and VIP/reopened status shape urgency and effort.

## Schema

| Column | Meaning |
| --- | --- |
| `ticket_id` | Unique synthetic ticket identifier. |
| `arrival_ts` | Ticket arrival timestamp in business hours. |
| `queue` | Functional support queue. |
| `priority` | Ticket urgency bucket from `P1` to `P4`. |
| `language` | Customer-facing language requirement. |
| `channel` | Intake channel such as portal, email, or chat. |
| `customer_tier` | Customer segment. |
| `is_vip` | VIP flag derived from enterprise cases. |
| `is_reopened` | Whether the ticket is a reopened issue. |
| `complexity` | Synthetic complexity label. |
| `estimated_effort_min` | Estimated work content in minutes. |
| `first_response_sla_min` | Target first-response time in business minutes. |
| `first_response_due_ts` | Explicit first-response deadline on the business calendar. |
| `resolution_sla_business_min` | Target resolution time in business minutes. |
| `resolution_due_ts` | Explicit resolution deadline on the business calendar. |
| `required_skill_tags` | Demand-side skill requirements encoded as pipe-delimited tags. |
| `ticket_subject` | Human-readable synthetic subject line for context only. |

## Generation Rules

- Random seed is fixed at `42` for reproducibility.
- The current generator creates `5` business days starting on Monday
  `2026-03-02 08:00:00`.
- Every P1--P4 first-response and resolution SLA uses business minutes on the
  Monday-to-Friday, `08:00` to `16:00` calendar. SLA clocks pause overnight
  and on weekends; for example, `1,440` business minutes is three business
  days, not 24 elapsed hours.
- Daily ticket volume varies between `80` and `120`, with Tuesday and Wednesday
  allowed to run slightly busier inside that range.
- Arrivals are continuous within the shift and mildly concentrated in late
  morning.
- Priority is sampled from queue-specific base weights, then adjusted by
  customer tier, VIP status, reopened status, and channel.
- Complexity is sampled from queue-specific weights, with extra mass on harder
  work for urgent or reopened tickets.
- Effort is sampled from queue- and complexity-specific ranges, with modest
  uplifts for some urgent and VIP cases.
- SLA deadlines skip weekends and non-business hours.
- Queue-specific complexity and effort ranges are used to keep records plausible.
- `required_skill_tags` use a compact role-aligned vocabulary:
  queue tags plus optional `de_language`, `enterprise_handling`,
  `api_specialist`, `incident_escalation`, `product_specialist`,
  and `integration_specialist`.

## Validation Expectations

The generated dataset should satisfy the following checks:

- No null or empty fields.
- Unique `ticket_id` values.
- Monotonically ordered `arrival_ts`.
- `arrival_ts <= first_response_due_ts <= resolution_due_ts` for every row.
- All timestamps fall inside the business calendar.
- Category mixes remain broadly consistent with configured generation weights.

## Interpretation Notes

- The current schedulers only read the subset needed for assignment:
  `ticket_id`, `arrival_ts`, `queue`, `priority`, `language`,
  `estimated_effort_min`, `first_response_due_ts`, and `resolution_due_ts`.
- `release_ts` and `duration_slots` are derived preprocessing fields, not CSV
  columns. `release_ts` is `arrival_ts` rounded up to the next 5-minute slot,
  and `duration_slots` is `estimated_effort_min` rounded up to 5-minute slots.
- `required_skill_tags` is kept in the dataset even before agent data exists
  because it makes later assignment logic explicit, even though the current
  schedulers do not use it as a hard constraint.
- `incident_escalation` marks `P1` work, while `product_specialist` and
  `integration_specialist` identify queue-specific complex work.
- `ticket_subject` is descriptive metadata, not a modeling key.
- The dataset is designed to be easy to explain in a portfolio setting rather
  than to mimic every edge case of a real support organization.
- The companion agent-supply file is documented in [agent_supply_dataset.md](agent_supply_dataset.md).
