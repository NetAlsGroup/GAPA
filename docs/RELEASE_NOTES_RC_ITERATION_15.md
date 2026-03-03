# Release Notes Draft (RC Iteration 15)

## Highlights
- Benchmark runner upgraded to release-grade metadata with `synthetic/live/real` sources.
- Regression gate now enforces mode-set consistency and remains schema-compatible.
- MNM forward path modularized into allocation/dispatch/stats for maintainability.
- Queue persistence failures are observable with structured error evidence (`task_id/op/error_type`).
- Soak and chaos harness added for deterministic stability verification.

## Compatibility
- Existing API endpoints and response schema remain backward-compatible.
- Additive telemetry fields only (`host_facts`, `config_snapshot`, `real_workload_meta`, recovery latency distribution).
- Mode fallback semantics unchanged: `MNM -> M -> SM -> S`.

## Known Risks
- Real workload benchmark path may show host-level jitter under heavy local contention.
- Full `unittest discover` includes legacy env-sensitive tests outside RC scope.

## Upgrade Notes
- No migration required for existing benchmark gate consumers.
- Optional: adopt `source=real` benchmark profile for release readiness evidence.
