# Promotion Candidates (Iteration 15)

## Candidate 1: Release Benchmark Tri-Source Pattern
- Source: `examples/run_perf_baseline.py` (`synthetic/live/real`)
- Why promote: standardizes release evidence collection with same gate schema.
- Target: `Code/.multi-agents` benchmark skill recipes.

## Candidate 2: Mode-Set Consistency Gate Rule
- Source: `tests/perf_regression_gate.py` missing/extra mode fail rows.
- Why promote: prevents silent gate bypass when mode coverage drifts.
- Target: shared QA matrix playbook.

## Candidate 3: Queue Persistence Error Observability Contract
- Source: `server/task_queue.py` (`task_id/op/error_type` + counters)
- Why promote: improves diagnosability without API breaking changes.
- Target: runtime resilience baseline patterns.

## Candidate 4: Deterministic Soak+Chaos Harness
- Source: `tests/soak_chaos_stability.py`
- Why promote: reusable bounded-failure validation scaffold.
- Target: multi-agent resilience validation toolkit.
