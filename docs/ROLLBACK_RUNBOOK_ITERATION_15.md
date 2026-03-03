# Rollback and Recovery Runbook (Iteration 15)

## Trigger Conditions
- Any P0 QA row fails.
- Regression gate blocks release (`passed=false`).
- Cross-platform matrix returns non-zero blocking issues.

## Recovery Workflow
1. Freeze release promotion and collect failing artifacts.
2. Re-run gates in `conda graph`:
   - `PYTHONPATH=. /Users/nakilea/anaconda3/envs/graph/bin/python -m unittest tests.test_perf_regression_gate tests.test_task_queue tests.test_distributed_workers`
   - `/Users/nakilea/anaconda3/envs/graph/bin/python .multi-agents/scripts/run_cross_platform_mode_gate.py`
3. If issue is benchmark-only, switch temporary evidence source to `synthetic` and retain `live/real` as non-blocking references.
4. If issue is runtime semantics, revert to last known-good commit and re-run step 2.
5. Publish rollback evidence in `.multi-agents/qa/` and update `docs/RC_CHECKLIST_ITERATION_15.md`.

## Validation Points
- `qa-cross-platform-mode-contract-state-consistency-iteration-04.json` must stay `passed=true`.
- `qa-release-candidate-closure-and-promotion-iteration-15.json` must have empty `blocking_issues`.

## Operator Diagnostics
- Transport/retry: `GET /api/transport/metrics`
- Queue durability: review `queue_persist_error` logs + `persistence_observability`
- MNM recovery: inspect `detailed_stats().recovery.latency_ms`
