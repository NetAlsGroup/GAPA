# Operations and QA

This page contains performance, stability, and release-oriented procedures. It is intentionally separated from the beginner onboarding path.

Docs hub:

- [README.md](README.md)

## Performance Baseline and Regression Gate

- Generate baseline metrics (synthetic):
  - `python examples/run_perf_baseline.py --profile small --source synthetic`
- Generate live baseline from a minimal runtime path:
  - `python examples/run_perf_baseline.py --profile small --source live --live-samples 48`
- Generate release-grade baseline with real workload sample:
  - `python examples/run_perf_baseline.py --profile release_small --source real --real-dataset ForestFire_n500 --real-generations 1 --real-pop-size 12 --real-runs 2`
- Re-run current metrics and compare with gate thresholds:
  - `python tests/perf_regression_gate.py --baseline <baseline.json> --current <current.json> --output <gate.json>`

Gate checks include mode-set consistency, throughput drop, latency increase, recovery latency increase, and remote failure rate delta for `S` / `SM` / `M` / `MNM`.

## Soak and Chaos Stability

- Run deterministic soak and chaos harness:
  - `python tests/soak_chaos_stability.py --iterations 80 --output .multi-agents/qa/qa-soak-and-chaos-stability-hardening-iteration-14.json`
- Keep release gate:
  - `python .multi-agents/scripts/run_cross_platform_mode_gate.py`

## MNM Communication Validation

- Generate before and after communication report:
  - `python tests/mnm_comm_iteration16_report.py --output .multi-agents/qa/mnm-communication-optimization-iteration-16.json`
- QA template result:
  - `.multi-agents/qa/qa-mnm-communication-algorithm-optimization-iteration-16.json`

## Release Candidate Package

- RC checklist: `docs/RC_CHECKLIST_ITERATION_15.md`
- Release notes: `docs/RELEASE_NOTES_RC_ITERATION_15.md`
- Rollback runbook: `docs/ROLLBACK_RUNBOOK_ITERATION_15.md`
- Promotion candidates: `docs/PROMOTION_CANDIDATES_ITERATION_15.md`

## Onboarding Consistency Gate

- Validate the maintained beginner path:
  - `python scripts/validate_onboarding_consistency.py`
