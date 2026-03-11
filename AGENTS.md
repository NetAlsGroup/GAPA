# GAPA AGENTS

Scope: `/Users/nakilea/Desktop/Code/GAPA`

Read this file after `/Users/nakilea/Desktop/Code/AGENTS.md`.

## Project Facts

- Project type: local Python-first graph optimization library and service surface
- Main domains:
  - core package: `gapa/`
  - examples: `examples/`
  - service/runtime: `server/`, `app.py`, `server_agent.py`, `web/`
- Local multi-agent assets live under `/Users/nakilea/Desktop/Code/GAPA/.multi-agents`
- Project kickoff file:
  - `/Users/nakilea/Desktop/Code/GAPA/.multi-agents/THREAD_KICKOFF.md`
- Directory migration guide:
  - `/Users/nakilea/Desktop/Code/GAPA/.multi-agents/DIRECTORY_MIGRATION.md`

## Thread Bootstrap

- If the user already states a role such as `Partner`, `CTO`, `TL`, or `execution`, route to that role first.
- If the role is missing, ask one bootstrap question first:
  - `本次线程角色是什么？Partner / CTO / TL / execution`
- Then read:
  - `/Users/nakilea/Desktop/Code/.multi-agents/ROLE_ROUTING.md`
  - `/Users/nakilea/Desktop/Code/GAPA/.multi-agents/THREAD_KICKOFF.md`
- For development/process guidance after role routing, load:
  - `/Users/nakilea/.codex/superpowers/.codex/superpowers-bootstrap.md`
- Use only the minimal matching Superpowers skill. Superpowers cannot override local role authority.

## Role Routing

- `Partner`: use root partner prompt plus repository/product framing from `README.md`.
- `CTO`: use root CTO prompt plus architecture and operations material in project-local `.multi-agents`.
- `TL`: use root TL prompt plus GAPA registry/orchestration/task packet context.
- `execution`: use the project kickoff chain and active task packet with minimal preload.

## Required Local Reads

Choose only what the task needs:

- Repository overview:
  - `/Users/nakilea/Desktop/Code/GAPA/README.md`
- Governance core:
  - preferred normalized design path: `/Users/nakilea/Desktop/Code/GAPA/.multi-agents/design/`
  - preferred normalized memory path: `/Users/nakilea/Desktop/Code/GAPA/.multi-agents/memory/`
  - `/Users/nakilea/Desktop/Code/GAPA/.multi-agents/README.md`
  - `/Users/nakilea/Desktop/Code/GAPA/.multi-agents/GAPA_BLUEPRINT.md`
  - `/Users/nakilea/Desktop/Code/GAPA/.multi-agents/ORCHESTRATION.md`
  - `/Users/nakilea/Desktop/Code/GAPA/.multi-agents/AGENT_REGISTRY.md`
- Runtime policy:
  - preferred normalized path: `/Users/nakilea/Desktop/Code/GAPA/.multi-agents/runtime/policy/`
  - `/Users/nakilea/Desktop/Code/GAPA/.multi-agents/runtime/workspace-policy.yaml`
  - `/Users/nakilea/Desktop/Code/GAPA/.multi-agents/runtime/sandbox-policy.yaml`
  - `/Users/nakilea/Desktop/Code/GAPA/.multi-agents/runtime/test-policy.yaml`
  - `/Users/nakilea/Desktop/Code/GAPA/.multi-agents/runtime/delivery-policy.yaml`
- Active task packet:
  - preferred normalized path: `/Users/nakilea/Desktop/Code/GAPA/.multi-agents/tasks/packets/`
  - `/Users/nakilea/Desktop/Code/GAPA/.multi-agents/tasks/<active-task>.md`

## Execution Guardrails

- Treat GAPA primarily as a Python library with optional service/web layers.
- Do not widen preload or staffing beyond the active task packet, especially when token cost is marked `moderate` or `heavy`.
- If a task affects runtime, service, or web paths, verify the corresponding execution mode instead of assuming library-only behavior.
- If direction is underdetermined, return to TL rather than inventing orchestration assumptions.

## Common Commands

- Editable install:
  - `cd /Users/nakilea/Desktop/Code/GAPA && pip install -e .`
- Example validation:
  - `cd /Users/nakilea/Desktop/Code/GAPA && python examples/api/workflow.py`
- Governance validation:
  - `cd /Users/nakilea/Desktop/Code/GAPA && python3 .multi-agents/scripts/validate_governance_pack.py`
- Package tests:
  - `cd /Users/nakilea/Desktop/Code/GAPA && pytest`

## Editing Rules

- Preserve the unified `DataLoader / Workflow / Monitor / ResourceManager` user-facing model unless the task explicitly changes it.
- Keep examples, docs, and runtime behavior aligned when changing public interfaces.
- Update project-local changelog or task evidence when the task packet requires governance closeout.
- New `.multi-agents` assets should follow the normalized `design / runtime / tasks / memory` split.
