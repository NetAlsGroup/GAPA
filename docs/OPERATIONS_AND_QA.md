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
- Compare the generated baselines with your own release thresholds in your local validation workflow.

Public documentation keeps the baseline generation path. Internal regression gates, private release thresholds, and governance automation are maintained outside the open-source workflow.

## Soak and Chaos Stability

- For public validation, use the maintained API and algorithm examples as the stable smoke path:
  - `python examples/api/workflow.py`
  - `python examples/algorithms/CND/sixdst.py`

Long-run soak, chaos, and private release gates remain part of the internal delivery workflow and are not required for open-source usage.

## MNM Communication Validation

For MNM validation in the public workflow, start from the local resource view and then move to the documented distributed setup:

- `python examples/api/resource_manager.py`
- `python server_agent.py`

Release-grade MNM communication benchmarking and QA packaging remain internal.

## Release Candidate Package

- RC checklist: `docs/RC_CHECKLIST_ITERATION_15.md`
- Release notes: `docs/RELEASE_NOTES_RC_ITERATION_15.md`
- Rollback runbook: `docs/ROLLBACK_RUNBOOK_ITERATION_15.md`
- Promotion candidates: `docs/PROMOTION_CANDIDATES_ITERATION_15.md`

## Onboarding Consistency Gate

The maintained beginner path is now defined by:

- `README.md`
- `examples/api/`
- `examples/algorithms/`

Internal onboarding consistency gates are no longer part of the public repository workflow.
