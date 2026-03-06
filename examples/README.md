# Examples

Use the examples in this order.

## 1. Core API

- `api/data_loader.py`
  - inspect the public dataset registry and load a dataset by name
- `api/workflow.py`
  - run a minimal local workflow with `Workflow`
- `api/monitor.py`
  - inspect `status()`, `result()`, and `report()`
- `api/resource_manager.py`
  - query resources and request a strategy plan
- `api/algorithm.py`
  - pass a user-defined `Algorithm` subclass into `Workflow`

## 2. Advanced and compatibility scripts

- `quickstart_local.py`
  - closest script equivalent to the package demo path
- `custom_algorithm_minimal.py`
  - existing minimal custom algorithm path
- `advanced/remote_single_server.py`
  - remote `s` / `sm` / `m`
- `advanced/mnm_multi_node.py`
  - heterogeneous MNM
- `run_sixdst.py`
  - compatibility execution entry
- `resource_scheduler.py`
  - resource query and lock management
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
