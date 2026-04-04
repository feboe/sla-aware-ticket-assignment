# SLA-Aware Ticket Assignment

Synthetic service-operations data for a ticket-assignment optimization project.

The project currently separates the problem into two datasets:
- `data/ticket_assignment_tickets.csv` for demand-side ticket arrivals
- `data/agents.csv` for the fixed agent roster

## Current Scope
- Generate a deterministic synthetic ticket dataset with priorities, effort estimates, and SLA deadlines.
- Keep agent supply in a separate hand-authored CSV with queue permissions, language coverage, and priority scope.
- Validate both data files with lightweight unit tests.
- Provide a simple greedy baseline as a first online benchmark.
- Provide a same-day look-ahead greedy benchmark to compare myopic dispatch
  against a stronger heuristic that can reserve future slots.
- Provide a current-slot OR scheduler that re-optimizes every 15 minutes and
  commits only start-now assignments.

## Generate Tickets
```powershell
python scripts\generate_ticket_assignment_data.py
```

## Validate Data Contracts
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

The metrics JSON now uses a compact overview layer plus nested `scheduled`,
`backlog`, and `agent_utilization` sections for the detailed breakdown.

## Run The Look-Ahead Greedy Benchmark
```powershell
python scripts\run_lookahead_greedy.py
```

This writes:
- `results/lookahead_greedy_schedule.csv`
- `results/lookahead_greedy_metrics.json`

The look-ahead heuristic keeps the greedy ticket ordering but can reserve the
earliest feasible future slot later the same day. It is intentionally still a
simple reservation heuristic rather than a re-optimizing planner.

## Run The OR Scheduler
```powershell
python scripts\run_or_scheduler.py
```

This writes:
- `results/or_scheduler_schedule.csv`
- `results/or_scheduler_metrics.json`

The OR scheduler re-solves a current-slot CP-SAT dispatch model every 15
minutes. It only considers tickets that are already open, commits only the
assignments that start in the current slot, and leaves unresolved tickets in
backlog for later runs.

## Files
- [docs/ticket_demand_dataset.md](docs/ticket_demand_dataset.md): ticket-demand schema and generation rules
- [docs/agent_supply_dataset.md](docs/agent_supply_dataset.md): fixed agent-roster schema and role design
