# Examples

Use the examples in this order.

## 1. First script

- `quickstart_local.py`
  - closest script equivalent to `python -m gapa demo`
  - uses built-in small graphs and no repo dataset

## 2. Minimal customization

- `custom_algorithm_minimal.py`
  - smallest custom algorithm wrapper path
  - runs a user-defined algorithm through `Workflow`

## 3. Advanced execution

- `advanced/remote_single_server.py`
  - recommended named entry for remote `s` / `sm` / `m`
- `advanced/mnm_multi_node.py`
  - recommended named entry for heterogeneous MNM
- `advanced/README.md`
  - index for remote, queue, lock, reporting, and baseline scripts

## Compatibility paths

These existing scripts are kept for compatibility and power users:

- `run_sixdst.py`
  - single-node (`s` / `sm` / `m`) and MNM execution
- `resource_scheduler.py`
  - server/resource query, lock/unlock, strategy plan
- `run_analysis_queue.py`
  - queue-aware remote submit and polling
- `run_lock_keepalive.py`
  - lock renew / release keepalive flow
- `run_report_export.py`
  - monitor export and reporting
- `run_trends.py`
  - trend aggregation for `results/run_reports.jsonl`
- `run_perf_baseline.py`
  - synthetic / live / real performance baseline generation
