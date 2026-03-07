# Examples

Public examples are intentionally kept minimal. Use them in this order.

These examples are source-tree examples. Run `pip install -e .` once before executing files under `examples/`.

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

## 2. Built-in algorithms

- `algorithms/CND/sixdst.py`
- `algorithms/CND/cutoff.py`
- `algorithms/CND/tde.py`
- `algorithms/CDA/cgn.py`
- `algorithms/CDA/qattack.py`
- `algorithms/CDA/cda_eda.py`
- `algorithms/NCA/gani.py`
- `algorithms/NCA/nca_ga.py`
- `algorithms/LPA/lpa_eda.py`
- `algorithms/LPA/lpa_ga.py`

## 3. Advanced usage

Advanced service, remote runtime, and compatibility guidance has been moved out of the public examples surface.

- advanced docs: `docs/ADVANCED_USAGE.md`
- operations and QA: `docs/OPERATIONS_AND_QA.md`
- remote entry points: `app.py`, `server_agent.py`, `server/`
- legacy script-first references: `old_examples/`
