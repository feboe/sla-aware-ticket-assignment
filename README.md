# SLA-Aware Ticket Assignment

This project benchmarks online dispatch policies for a synthetic B2B support
operation where tickets differ by queue, language, effort, and SLA priority,
and agents differ by permissions, coverage, daily capacity, and shifts. The
benchmark asks a practical operations question: which dispatch policy gives the
best trade-off between throughput and protecting urgent tickets?

Three policies are compared on the same replay environment: a myopic greedy
baseline, a same-day look-ahead greedy heuristic, and a rolling CP-SAT OR
scheduler. In the current 5-minute setup, `lookahead_greedy` achieves the
highest throughput, while `or_scheduler` is the recommended business-facing
option when P1/P2 protection matters most. The benchmark is intentionally
overloaded: raw demand is `19,439` minutes against `19,200` minutes of 5-day
agent capacity, and the active 5-minute slotted workload rises to `20,460`
minutes.

For a deeper look at the OR approach, start with
[Problem Overview](docs/problem_formulation.md#problem-overview),
[Rolling Policy](docs/problem_formulation.md#rolling-policy), and
[Objective](docs/problem_formulation.md#objective) in the problem formulation.

## Key Result
- 5-minute slots are the practical default: they preserve most of the timing
  detail without turning the benchmark into a one-minute dispatch simulation.
- `lookahead_greedy` maximizes scheduled volume and overall utilization.
- `or_scheduler` is the strongest policy when the evaluation puts P1/P2 urgency
  ahead of raw throughput.

## Benchmark Snapshot (5-Minute Slots)

Not all tickets can be scheduled within the available agent capacity, so policy
quality should be judged by how well urgent work is protected under overload.

| Policy | Scheduled | Backlog | P1 Backlog | P2 Backlog | P1 FR Tardiness (min) | P2 FR Tardiness (min) | Utilization | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Greedy | 442 | 69 | 0 | 1 | 175.74 | 3376.41 | 93.57% | Urgent-ticket handling is solid, but overall throughput is weakest |
| Look-Ahead Greedy | 476 | 35 | 4 | 5 | 15644.30 | 34697.51 | 97.66% | Best throughput, but clearly weaker on urgent-ticket protection |
| OR Scheduler | 452 | 59 | 0 | 1 | 268.99 | 3240.13 | 94.17% | Best business-facing trade-off when P1/P2 matter most |

Overall throughput favors look-ahead greedy, but a business-facing assessment
should rank urgent-ticket protection first. That is why this project recommends
the OR scheduler as the stronger dispatch story: its weighting scheme
explicitly prioritizes P1/P2 tickets. In an overloaded system,
that prioritization is the core operational decision, not a secondary
preference. For the mathematical rationale behind that choice, see
[Priority Weights](docs/problem_formulation.md#priority-weights) and
[Objective](docs/problem_formulation.md#objective).

The OR scheduler is also operationally lightweight in this setup: it solved 342
rolling decisions at 5-minute granularity with an average solve time of 0.0017
seconds, and all solves finished with `OPTIMAL` status.

## Why 5-Minute Slots

The current benchmark uses 5-minute decision slots. This is the practical
compromise between fidelity and realism: 15-minute slots were too coarse for
the ticket arrivals and effort distribution, while 1-minute slots are more
detailed than most real support dispatch loops need. For the slotting logic
behind `release_ts`, `duration_slots`, and the remaining-day horizon, see
[Time Structure](docs/problem_formulation.md#time-structure).

## Benchmark Setup

The repo separates synthetic demand generation, fixed agent supply, replay
policies, and evaluation outputs so the methods stay easy to compare.

- `data/tickets.csv`: synthetic demand-side arrivals with SLA deadlines
- `data/agents.csv`: fixed hand-authored agent roster
- `scripts/generate_ticket_assignment_data.py`: deterministic ticket generator
- `src/greedy_baseline.py`: online myopic dispatcher
- `src/lookahead_greedy.py`: same-day earliest-fit reservation heuristic
- `src/or_scheduler.py`: rolling current-slot OR scheduler built on CP-SAT
- `src/preprocessing.py` and `src/evaluation.py`: shared slotting, feasibility,
  schedule writing, and metrics aggregation utilities

If you want the dataset design details behind the benchmark, start with
[Ticket Demand Dataset](docs/ticket_demand_dataset.md) and
[Agent Supply Dataset](docs/agent_supply_dataset.md).

## Benchmark Policies
- `greedy_baseline`: only considers tickets that are already open and agents
  that are idle now. It assigns tickets in deterministic urgency order and does
  not reserve future capacity.
- `lookahead_greedy`: keeps the same urgency ordering, but can reserve the
  earliest feasible future slot later on the same business day.
- `or_scheduler`: re-solves a current-slot CP-SAT model every 5 minutes,
  considers only tickets already open at that solve, and commits only starts in
  the current slot before re-optimizing later.

## Generate Tickets
```powershell
python scripts\generate_ticket_assignment_data.py
```

This writes `data/tickets.csv`.

## Validate The Project
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## Run The Greedy Baseline
```powershell
python scripts\run_greedy_baseline.py
```

This writes:
- `results/greedy_baseline_schedule.csv`
- `results/greedy_baseline_metrics.json`

## Run The Look-Ahead Greedy Benchmark
```powershell
python scripts\run_lookahead_greedy.py
```

This writes:
- `results/lookahead_greedy_schedule.csv`
- `results/lookahead_greedy_metrics.json`

## Run The OR Scheduler
```powershell
python scripts\run_or_scheduler.py
```

This writes:
- `results/or_scheduler_schedule.csv`
- `results/or_scheduler_metrics.json`

The OR scheduler uses a rolling current-slot objective with weighted
first-response tardiness, weighted resolution tardiness, and a small direct
backlog term. It is not a full future-slot planner.

## Outputs

All three runners write:
- one schedule CSV in `results/`
- one metrics JSON in `results/`

The final replay schedule uses the shared schema from `src/evaluation.py` and
persists only:
- `scheduled`
- `backlog_end`

All three metrics JSON files share the same core structure:
- top-level replay summary fields such as `replay_business_days`,
  `slot_minutes`, `horizon_start_ts`, `horizon_end_ts`, `total_tickets`,
  `scheduled_tickets`, and `tickets_in_backlog`
- nested `scheduled` and `backlog` sections with the same priority breakdown
- per-agent `agent_utilization` plus `overall_agent_utilization`

The OR scheduler adds solver-specific diagnostics:
- `solve_call_count`
- `avg_solve_time_sec`
- `solver_status_counts`

## Documentation
- [Problem Formulation](docs/problem_formulation.md)
- [Ticket Demand Dataset](docs/ticket_demand_dataset.md)
- [Agent Supply Dataset](docs/agent_supply_dataset.md)
