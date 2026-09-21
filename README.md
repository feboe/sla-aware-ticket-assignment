# SLA-Aware Ticket Assignment

This project benchmarks online dispatch policies for a synthetic B2B support
operation where tickets differ by queue, language, effort, and SLA priority,
and agents differ by permissions, coverage, daily capacity, and shifts. The
benchmark asks a practical operations question: which dispatch policy gives the
best trade-off between throughput and protecting urgent tickets?

Three policies are compared on the same replay environment: a myopic greedy
baseline, a same-day look-ahead greedy heuristic, and a rolling CP-SAT OR
scheduler. In the current 5-minute setup, `lookahead_greedy` achieves the
highest throughput, `greedy_baseline` is slightly best on P1 first-response
tardiness, and `or_scheduler` improves total and P2 first-response tardiness
versus greedy while keeping P1 performance close. The benchmark is
intentionally overloaded: raw demand is `19,439` minutes against `19,200`
minutes of 5-day agent capacity, and the active 5-minute slotted workload rises
to `20,460` minutes.

For a deeper look at the OR approach, start with
[Problem Overview](docs/problem_formulation.md#problem-overview),
[Rolling Policy](docs/problem_formulation.md#rolling-policy), and
[Objective](docs/problem_formulation.md#objective) in the problem formulation.

## Why This Matters

- This project turns a realistic operations problem into a reproducible OR
  benchmark: overloaded support demand, limited agent supply, and competing SLA
  priorities.
- It compares simple online heuristics against a rolling CP-SAT scheduler in
  the same replay environment, so the trade-offs are easy to explain and verify.
- The key lesson is that maximizing throughput is not the same as protecting the
  most urgent tickets. In this benchmark, the OR scheduler is useful because
  it makes that business trade-off explicit instead of relying on raw volume
  alone.
- As a first OR portfolio project, the repo is designed to show the full loop:
  problem framing, synthetic data design, heuristic baselines, optimization
  modeling, and reproducible evaluation.

## Key Result

- 5-minute slots are the practical default: they preserve most of the timing
  detail without turning the benchmark into a one-minute dispatch simulation.
- Policy quality is assessed with SLA tardiness measured in business minutes,
  not elapsed calendar minutes.
- `lookahead_greedy` maximizes scheduled volume and overall utilization.
- `or_scheduler` provides the stronger SLA-aware compromise against greedy,
  even though it does not win every urgent-ticket metric.

## Benchmark Snapshot (5-Minute Slots)

Not all tickets can be scheduled within the available agent capacity, so policy
quality should be judged by how well urgent work is protected under overload.
All SLA metrics use business minutes on the Monday--Friday, 08:00--16:00 support
calendar; see [Time Structure](docs/problem_formulation.md#time-structure) for
the detailed time semantics.

| Policy | Scheduled | Backlog | Total First Response Tardiness (business min) | P1 First Response Tardiness (business min) | P2 First Response Tardiness (business min) | Utilization | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Greedy | 442 | 69 | 72,405.51 | 175.74 | 496.41 | 93.57% | Strongest narrow P1 handling, but weaker overall |
| Look-Ahead Greedy | 476 | 35 | 25,823.44 | 4,124.30 | 9,737.51 | 97.66% | Throughput winner, but clearly weaker on urgent-ticket protection |
| OR Scheduler | 450 | 61 | 54,713.56 | 219.02 | 207.18 | 93.98% | Better overall and P2 protection than greedy while keeping P1 close |

Resolution tardiness is secondary in the current benchmark: greedy and OR both
finish at 0.0 business minutes, while look-ahead reaches 980.97.

Overall throughput favors look-ahead greedy, but a business-facing assessment
should rank urgent-ticket protection first. Greedy is slightly best on P1,
look-ahead greedy is best on throughput, and the OR scheduler is the strongest
optimization-based compromise against greedy: it schedules 8 more tickets,
lowers total first-response tardiness from 72,405.51 to 54,713.56 business
minutes, and lowers P2 tardiness from 496.41 to 207.18 business minutes. For the
mathematical rationale behind that choice, see
[Priority Weights](docs/problem_formulation.md#priority-weights) and
[Objective](docs/problem_formulation.md#objective).

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
- `src/business_calendar.py`: shared business calendar and SLA-time utilities
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

## Setup

This repo targets Python `3.10+` and uses Linux shell commands as the primary
workflow. On Debian or Ubuntu, install `python3-venv` first if the `venv` module
is not already available.

```bash
git clone https://github.com/feboe/sla-aware-ticket-assignment.git
cd sla-aware-ticket-assignment
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -p "test_*.py"
```

If you want to reproduce the published benchmark outputs from a fresh clone,
run the data generator once and then execute the three benchmark scripts below.

## Generate Tickets

```bash
.venv/bin/python -m scripts.generate_ticket_assignment_data
```

This writes `data/tickets.csv`.

## Validate The Project

```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py"
```

## Run The Greedy Baseline

```bash
.venv/bin/python -m scripts.run_greedy_baseline
```

This writes:

- `results/greedy_baseline_schedule.csv`
- `results/greedy_baseline_metrics.json`

## Run The Look-Ahead Greedy Benchmark

```bash
.venv/bin/python -m scripts.run_lookahead_greedy
```

This writes:

- `results/lookahead_greedy_schedule.csv`
- `results/lookahead_greedy_metrics.json`

## Run The OR Scheduler

```bash
.venv/bin/python -m scripts.run_or_scheduler
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

## Limitations

- The OR scheduler is a rolling current-slot model, not a full-day planner. It
  decides what can start now and re-optimizes every 5 minutes.
- Future ticket arrivals are unknown inside any single solve, so the model makes decisions with current information only.
- Deferred work is penalized with a remaining-day horizon proxy, and end-of-run
  backlog metrics are measured at the final replay horizon end. Both use the
  shared business-time calendar; the local proxy is still not a full future-day
  schedule.
- `required_skill_tags` and agent `skill_tags` are documented in the datasets
  but are still metadata in the current version, not hard matching constraints.
- The benchmark is synthetic and portfolio-oriented by design: it aims to be
  explainable and reproducible rather than to capture every edge case of a real
  support organization.

## Documentation

- [Problem Formulation](docs/problem_formulation.md)
- [Ticket Demand Dataset](docs/ticket_demand_dataset.md)
- [Agent Supply Dataset](docs/agent_supply_dataset.md)
