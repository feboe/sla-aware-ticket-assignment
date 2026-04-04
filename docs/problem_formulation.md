# Problem Formulation

## Problem Overview

This project models a rolling support-ticket scheduler for a synthetic B2B
service operation. Tickets arrive over time and differ in queue, language,
priority, effort, and SLA targets. Agents differ in queue permissions,
language coverage, priority scope, daily capacity, and shift bounds.

The current OR model is a realistic current-slot scheduler. Every 15 minutes it
looks at the tickets that are already open, filters to agent-ticket pairs that
can legally start now, and decides which tickets to start immediately in the
current slot.

## Scope And Assumptions

The current model uses the following assumptions:

- Each ticket is processed by at most one agent.
- Once a ticket starts, it runs in consecutive 15-minute slots without preemption.
- A started ticket must finish within the assigned agent's same-day shift.
- Tickets not started in the current run remain in backlog and can be reconsidered later.
- First response is identified with the ticket's start time.
- Resolution is identified with the ticket's completion time.
- Hard eligibility is based on `queue_permissions`, `languages`, and `priority_scope`.
- `skill_tags` are retained as metadata but are not hard constraints in the current model.

## Rolling Policy

Operationally, the scheduler runs as follows:

1. Every 15 minutes, collect all tickets released up to the current slot.
2. Combine new arrivals with unresolved backlog.
3. Exclude agent-ticket pairs that violate queue, language, priority, occupancy,
   remaining daily capacity, or same-day completion rules.
4. Solve a current-slot CP-SAT assignment model.
5. Commit only the assignments that start now.
6. Carry unresolved tickets forward to the next rolling run.

Future ticket arrivals are unknown and are not modeled.

## Time Structure

Time is discretized into 15-minute slots.

For one scheduler run:

- `decision_ts` is the current rounded decision slot.
- all chosen assignments start at `decision_ts`
- `horizon_end_ts` is the latest shift end among agents on that day
- `horizon_slot_count` is the number of remaining 15-minute slots from
  `decision_ts` to `horizon_end_ts`

The solver is not a full-day planner. It chooses only start-now assignments, but
it still uses the remaining-day horizon as a proxy when penalizing backlog.

## Sets And Indices

- `I`: set of tickets open at the current decision slot
- `A`: set of agents
- `A_i`: feasible agents that can legally start ticket `i` now

Typical indices:

- `i in I`: ticket index
- `a in A`: agent index

## Parameters

### Ticket Parameters

- `q_i`: queue of ticket `i`
- `l_i`: language requirement of ticket `i`
- `pr_i`: priority of ticket `i`
- `p_i`: processing time of ticket `i` in 15-minute slots
- `d_i^{FR}`: first-response due timestamp of ticket `i`
- `d_i^{RES}`: resolution due timestamp of ticket `i`

### Agent Parameters

- `Q_a`: queues agent `a` may handle
- `L_a`: languages agent `a` supports
- `P_a`: priorities agent `a` may handle
- `s_a`: shift start time of agent `a`
- `f_a`: shift end time of agent `a`
- `c_a`: remaining daily capacity of agent `a` in minutes

### Priority Weights

The objective uses the following SLA weights:

- first-response weights: `P1=1000`, `P2=200`, `P3=40`, `P4=10`
- resolution weights: `P1=100`, `P2=20`, `P3=4`, `P4=1`

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

- `x_{i,a} in {0,1}`: equals `1` if ticket `i` starts now on agent `a`
- `b_i in {0,1}`: equals `1` if ticket `i` is deferred to backlog in the current run
- `u_i^{FR} in Z_{>=0}`: first-response tardiness of ticket `i`, measured in slots
- `u_i^{RES} in Z_{>=0}`: resolution tardiness of ticket `i`, measured in slots

There are no future-slot decision variables in the current model.

## Constraints

### 1. Assignment Or Backlog

Each open ticket is either started now on exactly one feasible agent or left in
backlog:

`sum_{a in A_i} x_{i,a} + b_i = 1    for all i in I`

### 2. One New Start Per Agent

An agent can start at most one new ticket in the current slot:

`sum_{i : a in A_i} x_{i,a} <= 1    for all a in A`

No additional within-model overlap constraints are needed for new starts,
because all chosen assignments start in the same slot and previously committed
occupancy has already been filtered in preprocessing.

### 3. First-Response Tardiness

Since any scheduled ticket starts immediately, the start offset is `0`. For a
backlogged ticket, the model uses the full remaining-day horizon as a proxy:

`u_i^{FR} >= H * b_i - delta_i^{FR}    for all i in I`

where:

- `H` is `horizon_slot_count`
- `delta_i^{FR}` is the first-response due timestamp converted to a slot offset
  relative to `decision_ts` using floor semantics

### 4. Resolution Tardiness

If ticket `i` starts now, its completion offset is `p_i`. If it is backlogged,
the same horizon proxy is added:

`u_i^{RES} >= p_i * sum_{a in A_i} x_{i,a} + H * b_i - delta_i^{RES}    for all i in I`

where `delta_i^{RES}` is the resolution due timestamp converted to a slot
offset relative to `decision_ts`.

Because tardiness variables are nonnegative integer variables, these inequalities
represent the positive-part tardiness approximation used by the solver.

## Objective

The current model minimizes weighted SLA tardiness only:

`min sum_{i in I} (w_i^{FR} * u_i^{FR} + w_i^{RES} * u_i^{RES})`

Important notes about the current objective:

- fairness is not included
- start-delay is not included
- there is no separate backlog-count penalty
- backlog still affects the objective indirectly through the tardiness proxy

This means the model values urgent tickets through explicit first-response and
resolution tardiness, while deferred tickets are penalized as if they slip
toward the end of the remaining workday.

## Data Mapping

The formulation maps directly to the current CSV inputs.

### Ticket-Demand File

From `data/tickets.csv`:

- `arrival_ts` -> arrival time
- `first_response_due_ts` -> first-response deadline
- `resolution_due_ts` -> resolution deadline
- `estimated_effort_min` -> processing effort
- `queue` -> queue requirement
- `language` -> language requirement
- `priority` -> urgency class

### Agent-Supply File

From `data/agents.csv`:

- `queue_permissions` -> eligible queues
- `languages` -> supported languages
- `priority_scope` -> eligible priorities
- `shift_start`, `shift_end` -> working-time bounds
- `capacity_min_per_day` -> daily capacity parameter

## Interpretation

This formulation is a realistic online dispatch model:

- it uses only currently available information
- it commits only current-slot starts
- it remains directly comparable to the greedy baselines
- it is small enough to solve quickly at every 15-minute decision point

At the same time, it is still an approximation of long-run backlog effects,
because deferred work is penalized through a remaining-day horizon proxy rather
than an explicitly modeled future schedule.
