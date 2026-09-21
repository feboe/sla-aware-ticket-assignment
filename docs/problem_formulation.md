# Problem Formulation

## Problem Overview

This project models a rolling support-ticket scheduler for a synthetic B2B
service operation. Tickets arrive over time and differ in queue, language,
priority, effort, and SLA targets. Agents differ in queue permissions,
language coverage, priority scope, daily capacity, and shift bounds.

The current OR model is a realistic current-slot scheduler. Every 5 minutes it
rounds the requested solve time up to the next 5-minute slot, looks at the
tickets that are already open at that rounded decision time, filters to
agent-ticket pairs that can legally start now, and decides which tickets to
start immediately in the current slot.

## Scope And Assumptions

The current model uses the following assumptions:

- Each ticket is processed by at most one agent.
- Once a ticket starts, it runs in consecutive 5-minute slots without preemption.
- A started ticket must finish within the assigned agent's same-day shift.
- Only tickets with `release_ts <= decision_ts` are modeled in a given solve.
- Tickets not started in the current run remain in backlog and can be
  reconsidered in later slots or later business days during replay.
- First response is identified with the ticket's start time.
- Resolution is identified with the ticket's completion time.
- Hard eligibility is based on `queue_permissions`, `languages`, and `priority_scope`.
- `required_skill_tags` and agent `skill_tags` are retained as metadata but are
  not hard constraints in the current model.

## Rolling Policy

Operationally, the scheduler runs as follows:

1. Every 5 minutes, request a solve and round that timestamp up to the next
   slot to obtain `decision_ts`.
2. Collect all tickets with `release_ts <= decision_ts`.
3. Combine new arrivals with unresolved backlog from previous runs.
4. Exclude agent-ticket pairs that violate queue, language, priority,
   occupancy, remaining daily capacity, or same-day completion rules.
5. Solve a current-slot CP-SAT assignment model.
6. Commit only the assignments that start at `decision_ts`.
7. Carry unresolved tickets forward to the next rolling run.

Within the full replay, backlog can carry across slots and across business
days. Future ticket arrivals are unknown and are not modeled inside any single
solve.

## Time Structure

Time is discretized into 5-minute slots.

All P1--P4 first-response and resolution SLA clocks use business minutes on a
single calendar: Monday through Friday, 08:00--16:00. Time overnight and on
weekends does not count toward a deadline or tardiness. The generator,
evaluation layer, and CP-SAT model use this same calendar. Thus, 24 business
hours is three business days in this eight-hour support operation, not 24
elapsed calendar hours.

The input CSV does not store slot-aligned arrivals. Preprocessing derives:

- `release_ts` by applying `ceil_to_slot(arrival_ts)`
- `duration_slots` by applying `ceil(estimated_effort_min / 5)` with a minimum
  of one slot

For one scheduler run:

- `requested_decision_ts` is the raw requested solve time
- `decision_ts` is the rounded current decision slot
- all chosen assignments start at `decision_ts`
- `horizon_end_ts` is the latest shift end among agents on that day
- `horizon_slot_count` is the number of remaining 5-minute business-time slots
  from `decision_ts` to `horizon_end_ts`

The solver is not a full-day planner. It chooses only start-now assignments,
but it still uses the remaining-day horizon as a proxy when penalizing backlog.

## Sets And Indices

- $I$: set of tickets open at the current decision slot
- $A$: set of agents
- $A_i$: feasible agents that can legally start ticket $i$ now

Typical indices:

- $i \in I$: ticket index
- $a \in A$: agent index

## Parameters

### Ticket Parameters

- $q_i$: queue of ticket $i$
- $l_i$: language requirement of ticket $i$
- $pr_i$: priority of ticket $i$
- $p_i$: processing time of ticket $i$ in 5-minute slots
- $d_i^{FR}$: first-response due timestamp of ticket $i$
- $d_i^{RES}$: resolution due timestamp of ticket $i$

### Agent Parameters

- $Q_a$: queues agent $a$ may handle
- $L_a$: languages agent $a$ supports
- $P_a$: priorities agent $a$ may handle
- $s_a$: shift start time of agent $a$
- $f_a$: shift end time of agent $a$
- $c_a$: remaining assignable minutes for agent $a$ in the current run after
  combining daily-capacity and remaining-shift limits

### Priority Weights

The objective uses the following SLA weights:

- $w_i^{FR} = W^{FR}(pr_i)$
- $w_i^{RES} = W^{RES}(pr_i)$
- $\lambda = 0.5$

with the following priority-to-weight mappings:

- $W^{FR}(P1)=1000,\; W^{FR}(P2)=200,\; W^{FR}(P3)=40,\; W^{FR}(P4)=10$
- $W^{RES}(P1)=100,\; W^{RES}(P2)=20,\; W^{RES}(P3)=4,\; W^{RES}(P4)=1$

These coefficients are manually chosen heuristic weights, not empirically
estimated cost parameters. They encode a clear business hierarchy in which
$P1 > P2 > P3 > P4$ is enforced strongly, first-response tardiness is valued
more heavily than resolution tardiness, and the backlog term remains only a
small secondary penalty.

### Derived Quantities

The tardiness constraints also use a few derived helper terms:

- $H$: remaining number of 5-minute business-time slots from `decision_ts` to
  `horizon_end_ts`
- $\delta_i^{FR}$: first-response due timestamp of ticket $i$ converted to a
  business-time slot offset relative to `decision_ts`, using floor semantics
  for non-slot-aligned timestamps
- $\delta_i^{RES}$: resolution due timestamp of ticket $i$ converted to a slot
  offset relative to `decision_ts`, also using business-time floor semantics

## Feasibility

A ticket-agent pair is feasible only if all of the following hold at the
current slot:

- the ticket is already released
- queue, language, and priority rules match
- the agent is not already occupied by previously committed work
- the ticket fits inside the agent's remaining daily capacity
- the ticket can finish before the agent's shift ends

These checks are handled before the CP-SAT model is built.

## Decision Variables

- $x_{i,a} \in \{0,1\}$ for $a \in A_i$: equals $1$ if ticket $i$ starts now on agent $a$
- $b_i \in \{0,1\}$: equals $1$ if ticket $i$ is deferred to backlog in the current run
- $u_i^{FR} \in \mathbb{Z}_{\ge 0}$: first-response tardiness of ticket $i$, measured in slots
- $u_i^{RES} \in \mathbb{Z}_{\ge 0}$: resolution tardiness of ticket $i$, measured in slots

There are no future-slot decision variables in the current model.

## Constraints

### 1. Assignment Or Backlog

Each open ticket is either started now on exactly one feasible agent or left in
backlog:

$$
\sum_{a \in A_i} x_{i,a} + b_i = 1 \quad \forall i \in I
$$

### 2. One New Start Per Agent

An agent can start at most one new ticket in the current slot:

$$
\sum_{i \in I : a \in A_i} x_{i,a} \le 1 \quad \forall a \in A
$$

No additional within-model overlap constraints are needed for new starts,
because all chosen assignments start in the same slot and previously committed
occupancy has already been filtered in preprocessing.

### 3. First-Response Tardiness

The first-response tardiness variable is linked to the decision as follows:

$$
u_i^{FR} \ge H b_i - \delta_i^{FR} \quad \forall i \in I
$$

This is easiest to read as two cases:

- if ticket $i$ is scheduled now, first response happens immediately at
  `decision_ts`, so the modeled first-response offset is $0$
- if ticket $i$ is deferred, the model uses $H$ as a proxy and penalizes the
  ticket as if first response slips to the end of the remaining day

Because $u_i^{FR} \in \mathbb{Z}_{\ge 0}$ and appears in the objective, it
settles at the positive-part first-response tardiness implied by that choice.

### 4. Resolution Tardiness

The resolution tardiness variable is linked to the decision as follows:

$$
u_i^{RES} \ge p_i \sum_{a \in A_i} x_{i,a} + H b_i - \delta_i^{RES} \quad \forall i \in I
$$

This also has two cases:

- if ticket $i$ is scheduled now, completion happens after $p_i$ slots, so the
  modeled resolution offset is $p_i$
- if ticket $i$ is deferred, the same remaining-day proxy $H$ is used

Under the current approximation, backlog behaves like "resolved by horizon
end," not "starts at horizon end and then runs for $p_i$."

Because $u_i^{RES} \in \mathbb{Z}_{\ge 0}$ and appears in the objective, it
settles at the positive-part resolution tardiness implied by that choice.

## Objective

The current model minimizes weighted first-response tardiness, weighted
resolution tardiness, and a small direct backlog penalty:

$$
\min \sum_{i \in I} \left( w_i^{FR} u_i^{FR} + w_i^{RES} u_i^{RES} + \lambda b_i \right)
$$

Important notes about the current objective:

- fairness is not included
- there is no future-slot start-delay term
- deferred tickets are penalized twice: through the remaining-day tardiness
  proxy and through the small direct backlog term

This means the model values urgent tickets through explicit first-response and
resolution tardiness, while still slightly preferring an immediate feasible
start over an otherwise equal deferral.

The objective should therefore be read as a compact encoding of business
priorities under overload, not as a calibrated estimate of real monetary cost.

## Data Mapping

The formulation maps directly to the current CSV inputs.

### Ticket-Demand File

From `data/tickets.csv`:

- `arrival_ts` -> raw arrival time
- `release_ts` -> derived by rounding `arrival_ts` up to the next slot
- `first_response_due_ts` -> first-response deadline
- `resolution_due_ts` -> resolution deadline
- `estimated_effort_min` -> processing effort
- `duration_slots` -> derived slot length
- `queue` -> queue requirement
- `language` -> language requirement
- `priority` -> urgency class

The CSV also contains descriptive fields such as `channel`, `customer_tier`,
`complexity`, `required_skill_tags`, and `ticket_subject`, but the current OR
scheduler does not use them as hard model inputs.

### Agent-Supply File

From `data/agents.csv`:

- `queue_permissions` -> eligible queues
- `languages` -> supported languages
- `priority_scope` -> eligible priorities
- `shift_start`, `shift_end` -> working-time bounds
- `capacity_min_per_day` -> daily capacity parameter
- `scarce_resource` -> metadata used by the greedy heuristics, not by the
  current OR scheduler

The CSV also contains `agent_role` and `skill_tags` for documentation and role
design, but the current OR scheduler does not treat them as hard constraints.

## Interpretation

This formulation is a realistic online dispatch model:

- it uses only currently available information
- it commits only current-slot starts
- it remains directly comparable to the greedy baselines
- it is small enough to solve quickly at every 5-minute decision point

At the same time, it is still an approximation of long-run backlog effects,
because deferred work is penalized through a remaining-day horizon proxy plus a
small direct backlog term rather than an explicitly modeled future schedule.
