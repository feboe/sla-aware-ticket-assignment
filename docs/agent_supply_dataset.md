# Agent Supply Dataset

## Use Case

This file defines a fixed support-team roster that can be paired with the ticket-demand dataset for later assignment and optimization work.

The agent roster is intentionally stored as a separate CSV so that supply and demand remain distinct inputs.

## File

- `data/agents.csv`

## Schema

| Column | Meaning |
| --- | --- |
| `agent_id` | Unique agent identifier. |
| `agent_role` | High-level team role. |
| `languages` | Supported customer languages, pipe-delimited. |
| `queue_permissions` | Queues the agent can own, pipe-delimited. |
| `priority_scope` | Priorities the agent is expected to handle, pipe-delimited. |
| `skill_tags` | Agent-side capability tags aligned with the ticket vocabulary. |
| `shift_start` | Daily shift start time. |
| `shift_end` | Daily shift end time. |
| `capacity_min_per_day` | Daily work capacity in minutes. |
| `scarce_resource` | Flag for intentionally scarce escalation capacity. |

## Team Design

- `3` generalists:
  bilingual (`EN|DE`), focused on `Account Access`, `Billing`, and standard `Product Support`, suitable for `P2-P4`
- `2` product specialists:
  dedicated to `Product Support`, both able to handle `P1-P4`, one bilingual and one English-only
- `2` integration specialists:
  dedicated to `Integrations/API`, both able to handle `P1-P4`, one bilingual and one English-only
- `1` senior incident coordinator:
  bilingual, cross-queue escalation resource focused on `P1/P2`, marked as scarce

## Interpretation Notes

- Queue permissions, languages, and priority scope are the main hard feasibility fields for the first version.
- `skill_tags` are still useful metadata, but they should not be treated as strict matching rules until the modeling layer needs them.
- The current role-aligned skill vocabulary is:
  `account_access`, `billing`, `product_support`, `integrations_api`,
  `api_specialist`, `incident_escalation`, `product_specialist`,
  `integration_specialist`, `enterprise_handling`, and `de_language`.
- `incident_escalation` is reserved for the senior incident coordinator, while `product_specialist` and `integration_specialist` distinguish queue-specific specialist work.
- The bilingual integration specialist exists to cover the current German-language integration demand in the ticket dataset.
