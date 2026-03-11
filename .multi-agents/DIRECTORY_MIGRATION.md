# GAPA Multi-Agent Directory Migration

Target model:
- `design/`: blueprints, registry, orchestration, prompts, skills metadata
- `runtime/`: policy and runtime-state artifacts
- `tasks/`: packets, examples, handoffs
- `memory/`: logs, experience, role memory

Current migration policy:
- legacy paths remain valid
- new assets should use the normalized layout first
- physical moves must preserve existing project references

Preferred future mapping:
- `README.md`, `GAPA_BLUEPRINT.md`, `ORCHESTRATION.md`, `AGENT_REGISTRY.md` -> `design/`
- `prompts/` and `skills/` -> `design/`
- `runtime/*.yaml` -> `runtime/policy/`
- `tasks/task-*.md` -> `tasks/packets/`
- `handoffs/` -> `tasks/handoffs/`
- `tasks/TASK_TEMPLATE*_EXAMPLE.md` -> `tasks/examples/`
- `logs/` and `EXPERIENCE_SYNC.md` -> `memory/`
