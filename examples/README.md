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

## Queue-aware remote task submit
- `run_analysis_queue.py`
  - Start analysis with `queue_if_busy`
  - Poll `/api/analysis/status` and `/api/analysis/queue`

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
