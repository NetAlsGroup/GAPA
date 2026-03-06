#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CheckRow:
    id: str
    status: str
    notes: str


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _contains_all(text: str, snippets: list[str]) -> bool:
    return all(snippet in text for snippet in snippets)


def run_checks() -> dict:
    rows: list[CheckRow] = []

    readme = _read(PROJECT_ROOT / "README.md")
    readme_zh = _read(PROJECT_ROOT / "README.zh-CN.md")
    cli_py = _read(PROJECT_ROOT / "gapa" / "cli.py")
    setup_py = _read(PROJECT_ROOT / "setup.py")
    examples_readme = _read(PROJECT_ROOT / "examples" / "README.md")
    docs_hub = _read(PROJECT_ROOT / "docs" / "README.md")
    docs_hub_zh = _read(PROJECT_ROOT / "docs" / "README.zh-CN.md")

    def add(row_id: str, ok: bool, notes: str) -> None:
        rows.append(CheckRow(id=row_id, status="pass" if ok else "fail", notes=notes))

    add(
        "ONBOARDING-README-ENTRY",
        _contains_all(readme, ["python -m gapa demo", "python -m gapa doctor", "docs/README.md"]),
        "README must expose demo, doctor, and docs hub links.",
    )
    add(
        "ONBOARDING-README-ZH-ENTRY",
        _contains_all(readme_zh, ["python -m gapa demo", "python -m gapa doctor", "docs/README.zh-CN.md"]),
        "Chinese README must expose demo, doctor, and docs hub links.",
    )
    add(
        "ONBOARDING-CLI-SURFACE",
        _contains_all(cli_py, ['"demo"', '"doctor"', '"smoke"']),
        "CLI must expose demo, doctor, and smoke commands.",
    )
    add(
        "ONBOARDING-SETUP-CONSOLE",
        "gapa=gapa.cli:main" in setup_py,
        "setup.py must publish the gapa console entrypoint.",
    )
    add(
        "ONBOARDING-EXAMPLES-TIERS",
        all(
            (PROJECT_ROOT / path).exists()
            for path in [
                "examples/quickstart_local.py",
                "examples/custom_algorithm_minimal.py",
                "examples/advanced/remote_single_server.py",
                "examples/advanced/mnm_multi_node.py",
            ]
        ),
        "Tiered example entrypoints must exist.",
    )
    add(
        "ONBOARDING-EXAMPLES-README",
        _contains_all(
            examples_readme,
            ["quickstart_local.py", "custom_algorithm_minimal.py", "advanced/remote_single_server.py"],
        ),
        "Examples README must describe the progressive order.",
    )
    add(
        "ONBOARDING-DOCS-HUB",
        _contains_all(docs_hub, ["ADVANCED_USAGE.md", "OPERATIONS_AND_QA.md", "RC_CHECKLIST_ITERATION_15.md"]),
        "Docs hub must link advanced usage, ops, and RC artifacts.",
    )
    add(
        "ONBOARDING-DOCS-HUB-ZH",
        _contains_all(
            docs_hub_zh,
            ["ADVANCED_USAGE.zh-CN.md", "OPERATIONS_AND_QA.zh-CN.md", "RC_CHECKLIST_ITERATION_15.md"],
        ),
        "Chinese docs hub must link advanced usage, ops, and RC artifacts.",
    )
    add(
        "ONBOARDING-INSTALL-TIERS",
        all((PROJECT_ROOT / path).exists() for path in [
            "requirements/core.txt",
            "requirements/full.txt",
            "requirements/attack.txt",
            "requirements/distributed.txt",
            "requirements/dev.txt",
        ]),
        "Requirement tier files must exist.",
    )

    passed = all(row.status == "pass" for row in rows)
    return {
        "passed": passed,
        "rows": [asdict(row) for row in rows],
        "blocking_issues": [row.id for row in rows if row.status != "pass"],
    }


def main() -> int:
    result = run_checks()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
