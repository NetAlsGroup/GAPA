# Examples

Script-first minimal examples (no Web UI dependency).

## Core GA run
- `run_sixdst.py`
  - Single-node (`s/sm/m`) and MNM execution
  - MNM lock/unlock workflow

## Resource and scheduler APIs
- `resource_scheduler.py`
  - Server/resource query
  - lock/unlock/lock-status
  - strategy plan and distributed plan
  - `transport-metrics` for retry/failure/degrade diagnostics

## Queue-aware remote task submit
- `run_analysis_queue.py`
  - Start analysis with `queue_if_busy`
  - Poll `/api/analysis/status` and `/api/analysis/queue`
  - Print normalized `mode_decision` (`requested_mode/selected_mode/degraded/reason/code`)
  - Supports `--checkpoint-ref` / `--retry-last` / `--schema-version`
  - Recognize terminal states: `completed/error/cancelled`

## Unstable Network Tips
- Prefer `queue_if_busy=true` and poll `analysis/status` instead of tight restart loops.
- Check `/api/transport/metrics` when remote nodes flap (failure rate / retries / recovery latency).

## Lock keepalive
- `run_lock_keepalive.py`
  - `lock_mnm()` -> `renew_mnm()` -> `unlock_servers()`

## Report export
- `run_report_export.py`
  - Run short workflow
  - Save report via `monitor.save_report()`
  - Print `monitor.run_trends()`

## Trend summary only
- `run_trends.py`
  - Aggregate `results/run_reports.jsonl`

## Benchmark and release validation
- `run_perf_baseline.py`
  - Synthetic baseline: `python examples/run_perf_baseline.py --profile small --source synthetic`
  - Live minimal path: `python examples/run_perf_baseline.py --profile small --source live --live-samples 48`
  - Real workload sample: `python examples/run_perf_baseline.py --profile release_small --source real --real-dataset ForestFire_n500 --real-generations 1 --real-pop-size 12 --real-runs 2`

## Soak and chaos harness
- Deterministic stability harness (test-side utility):
  - `python tests/soak_chaos_stability.py --iterations 80 --output .multi-agents/qa/qa-soak-and-chaos-stability-hardening-iteration-14.json`
