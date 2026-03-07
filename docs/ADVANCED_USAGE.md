# Advanced Usage

This page collects service, remote runtime, and advanced integration guidance that is intentionally kept out of the beginner onboarding path.

Docs hub:

- [README.md](README.md)

## Public Entry Points

The maintained public example surface is:

- `examples/api/`
- `examples/algorithms/`

Legacy script-first references are preserved under `old_examples/` for compatibility only.

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

- `.env` / `.env.example`: service host, remote endpoints, and resource filters
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

1. Implement your wrapper in your own package or start from `examples/api/algorithm.py`.
2. Register the algorithm in `algorithms.json` with `entry`, `init_kwargs`, and `capabilities`.
3. Start runs via `Workflow`.
4. For remote or MNM execution, make sure both local `app.py` and remote `server_agent.py` can import the registered entry.
