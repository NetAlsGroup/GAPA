# Advanced Usage

This page collects script-first, service, and advanced runtime guidance that is intentionally kept out of the beginner onboarding path.

Docs hub:

- [README.md](README.md)

## Script Entry Points

Core examples live under `examples/`:

| File | Purpose |
|---|---|
| `examples/run_sixdst.py` | end-to-end execution in `s` / `sm` / `m` / `mnm` |
| `examples/resource_scheduler.py` | resource listing, lock, and strategy planning |
| `examples/run_lock_keepalive.py` | lock / renew / release keepalive flow |
| `examples/run_analysis_queue.py` | queue-based remote scheduling |
| `examples/run_report_export.py` | monitor export and reporting |
| `examples/run_trends.py` | run trend aggregation |
| `examples/sixdst_custom.py` | example custom algorithm wrapper |

Example commands:

```bash
python examples/run_sixdst.py --dataset ForestFire_n500 --mode s
python examples/run_sixdst.py --dataset ForestFire_n500 --mode m --server 6 --use-strategy-plan
python examples/run_sixdst.py --dataset ForestFire_n500 --mode mnm --servers "Server 6"
```

## Services

Start local orchestrator service:

```bash
python app.py
```

Start remote server agent on each compute host:

```bash
uvicorn server_agent:app --host 0.0.0.0 --port 7777
```

## Runtime Config

- `servers.json`: remote server inventory and endpoints
- `algorithms.json`: registry for generic and user-defined algorithms

Useful environment variables:

- `GAPA_MNM_MAX_WORKERS` (default: `4`)
- `GAPA_MNM_REFRESH_S` (default: `2.0`)
- `GAPA_MNM_MIN_CHUNK_SIZE` (default: `6`)
- `GAPA_MNM_COMM_WINDOW_ITERS` (default: `3`)
- `GAPA_MNM_FP16_MIN_ROWS` (default: `24`)
- `GAPA_RPC_COMPRESS_MIN_BYTES` (default: `2048`)
- `GAPA_RPC_COMPRESS_MIN_SAVING` (default: `0.05`)
- `GAPA_RPC_TORCH_LEGACY_SERIALIZATION` (default: `1`)

## Custom Algorithm Integration

For user-defined or generic algorithms, use `algorithms.json` as the registration entry point.

Recommended flow:

1. Implement your wrapper under `examples/` or your own package.
2. Register the algorithm in `algorithms.json` with `entry`, `init_kwargs`, and `capabilities`.
3. Start runs via `Workflow` or example scripts.
4. For remote or MNM execution, make sure both local `app.py` and remote `server_agent.py` can import the registered entry.
