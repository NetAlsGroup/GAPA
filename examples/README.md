# Examples

Public examples are intentionally kept minimal. Use them in this order.

These examples are source-tree examples. Run `pip install -e .` once before executing files under `examples/`.

## 1. Core API

- `api/data_loader.py`
  - inspect the public dataset registry and load a dataset by name
- `api/workflow.py`
  - run a local `s` / `sm` / `m` workflow with CLI-selectable algorithm, dataset, steps, and constructor kwargs
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

## 3. Remote API

- `remote/resource_manager.py`
  - inspect configured remote servers and request a remote strategy plan
- `remote/workflow.py`
  - run a remote M-mode workflow with CLI-selectable server id, devices, algorithm, dataset, and steps

## 4. Advanced API

- `advanced/mnm_workflow.py`
  - run an MNM workflow with resource locks, CLI-selectable servers/devices, timing summaries, and communication metrics
- `advanced/service_api.py`
  - smoke test the local `app.py` service API surface
- `advanced/resource_lock.py`
  - exercise lock, renew, release on the first online remote server

## 5. Advanced usage

Advanced service, remote runtime, and compatibility guidance has been moved out of the public examples surface.

- advanced docs: `docs/ADVANCED_USAGE.md`
- operations and QA: `docs/OPERATIONS_AND_QA.md`
- remote entry points: `app.py`, `server_agent.py`, `server/`
- legacy script-first references: `old_examples/`

## Common Commands

Local workflow:

```bash
python examples/api/workflow.py --mode m --algorithm SixDST --dataset yeast1 --steps 100 --pop-size 80
```

Remote M-mode workflow:

```bash
python examples/remote/workflow.py --algorithm CDAEDA --dataset karate --steps 100 --remote-devices 0
```

MNM workflow:

```bash
python examples/advanced/mnm_workflow.py \
  --algorithm SixDST \
  --dataset yeast1 \
  --steps 200 \
  --server-ids Node2,Node1 \
  --lock-devices Node2=0 \
  --lock-devices Node1=0
```
