from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "calosum"
DOCS_ROOT = REPO_ROOT / "docs"

REQUIRED_PATHS = [
    REPO_ROOT / "AGENTS.md",
    DOCS_ROOT / "index.md",
    DOCS_ROOT / "ARCHITECTURE.md",
    DOCS_ROOT / "PLANS.md",
    DOCS_ROOT / "QUALITY_SCORE.md",
    DOCS_ROOT / "RELIABILITY.md",
    DOCS_ROOT / "INFRASTRUCTURE.md",
    DOCS_ROOT / "production-roadmap.md",
    DOCS_ROOT / "references" / "harness-engineering.md",
    DOCS_ROOT / "design-docs" / "core-beliefs.md",
    DOCS_ROOT / "product-specs" / "calosum-system.md",
    DOCS_ROOT / "exec-plans" / "tech-debt-tracker.md",
]

REQUIRED_DOC_LINKS = [
    "docs/index.md",
    "docs/ARCHITECTURE.md",
    "docs/PLANS.md",
    "docs/QUALITY_SCORE.md",
    "docs/RELIABILITY.md",
    "docs/INFRASTRUCTURE.md",
]

PLAN_REQUIRED_HEADINGS = [
    "# ",
    "## Purpose",
    "## Scope",
    "## Validation",
    "## Progress",
    "## Decision Log",
]

MODULE_RULES: dict[str, set[str]] = {
    # SHARED
    "shared.types": set(),
    "shared.ports": {"shared.types", "domain.metacognition"},
    "shared.async_utils": set(),
    "shared.serialization": set(),

    # DOMAIN
    "domain.advanced_interfaces": {"shared.types", "domain.metacognition", "shared.ports"},
    "domain.agent_execution": {"shared.async_utils", "shared.ports", "shared.types"},
    "domain.bridge": {"shared.types"},
    "domain.event_bus": set(),
    "domain.right_hemisphere": {"shared.types"},
    "domain.left_hemisphere": {"shared.types"},
    "domain.runtime": {"domain.runtime_dsl", "shared.types"},
    "domain.runtime_dsl": {"shared.types"},
    "domain.memory": {"shared.types"},
    "domain.persistent_memory": {"domain.memory", "shared.serialization", "shared.types"},
    "domain.telemetry": {"shared.types"},
    "domain.metacognition": {"domain.bridge", "shared.types"},
    "domain.orchestrator": {
        "shared.async_utils",
        "domain.bridge",
        "domain.event_bus",
        "domain.agent_execution",
        "domain.left_hemisphere",
        "domain.memory",
        "domain.metacognition",
        "shared.ports",
        "domain.right_hemisphere",
        "domain.runtime",
        "domain.telemetry",
        "shared.types",
    },

    # BOOTSTRAP
    "bootstrap.settings": set(),
    "bootstrap.factory": {
        "adapters.action_runtime", 
        "adapters.llm_qwen", 
        "adapters.memory_qdrant",
        "adapters.right_hemisphere_hf",
        "adapters.text_embeddings",
        "domain.memory",
        "domain.orchestrator",
        "domain.persistent_memory",
        "domain.right_hemisphere",
        "bootstrap.settings",
        "domain.telemetry"
    },
    "bootstrap.cli": {
        "bootstrap.factory",
        "domain.metacognition",
        "shared.serialization",
        "bootstrap.settings",
        "domain.telemetry",
        "shared.types",
    },
    "bootstrap.api": {
        "bootstrap.factory",
        "bootstrap.settings",
        "shared.serialization",
        "shared.types",
    },
    "bootstrap.__main__": {"bootstrap.cli"},

    # ADAPTERS
    "adapters.action_runtime": {"shared.async_utils", "shared.types"},
    "adapters.llm_payloads": {"shared.types"},
    "adapters.llm_qwen": {"adapters.llm_payloads", "shared.async_utils", "shared.types"},
    "adapters.memory_qdrant": {"adapters.text_embeddings", "shared.async_utils", "shared.types", "domain.memory"},
    "adapters.right_hemisphere_hf": {"shared.types"},
    "adapters.text_embeddings": {"shared.async_utils"},
    "adapters.night_trainer": set(),

    # ROOT
    "harness_checks": set(),
    "__init__": {
        "bootstrap.factory",
        "domain.bridge",
        "domain.left_hemisphere",
        "domain.memory",
        "domain.metacognition",
        "domain.orchestrator",
        "domain.persistent_memory",
        "shared.ports",
        "domain.right_hemisphere",
        "domain.runtime",
        "shared.serialization",
        "bootstrap.settings",
        "domain.telemetry",
        "shared.types",
    },
}

MAX_MODULE_LINES = 360


@dataclass(slots=True)
class HarnessIssue:
    code: str
    message: str
    path: str | None = None


@dataclass(slots=True)
class HarnessReport:
    passed: bool
    issues: list[HarnessIssue] = field(default_factory=list)


def run_harness_checks(repo_root: Path | None = None) -> HarnessReport:
    root = repo_root or REPO_ROOT
    issues: list[HarnessIssue] = []
    issues.extend(_check_required_paths(root))
    issues.extend(_check_agents_map(root))
    issues.extend(_check_docs_index(root))
    issues.extend(_check_plan_files(root))
    issues.extend(_check_module_sizes(root))
    issues.extend(_check_import_boundaries(root))
    return HarnessReport(passed=not issues, issues=issues)


def main() -> int:
    report = run_harness_checks()
    if report.passed:
        print("Harness checks passed.")
        return 0

    print("Harness checks failed:")
    for issue in report.issues:
        location = f" [{issue.path}]" if issue.path else ""
        print(f"- {issue.code}{location}: {issue.message}")
    return 1


def _check_required_paths(root: Path) -> list[HarnessIssue]:
    issues: list[HarnessIssue] = []
    for path in REQUIRED_PATHS:
        if not (root / path.relative_to(REPO_ROOT)).exists():
            issues.append(
                HarnessIssue(
                    code="missing_required_path",
                    message="required harness artifact is missing",
                    path=str(path.relative_to(REPO_ROOT)),
                )
            )
    return issues


def _check_agents_map(root: Path) -> list[HarnessIssue]:
    path = root / "AGENTS.md"
    text = path.read_text(encoding="utf-8")
    issues: list[HarnessIssue] = []
    if len(text.splitlines()) > 120:
        issues.append(
            HarnessIssue(
                code="agents_too_long",
                message="AGENTS.md should remain a short map, not a manual",
                path="AGENTS.md",
            )
        )
    for link in REQUIRED_DOC_LINKS:
        if link not in text:
            issues.append(
                HarnessIssue(
                    code="agents_missing_link",
                    message=f"missing required link to {link}",
                    path="AGENTS.md",
                )
            )
    return issues


def _check_docs_index(root: Path) -> list[HarnessIssue]:
    path = root / "docs" / "index.md"
    text = path.read_text(encoding="utf-8")
    required_refs = [
        "ARCHITECTURE.md",
        "PLANS.md",
        "QUALITY_SCORE.md",
        "RELIABILITY.md",
        "INFRASTRUCTURE.md",
        "production-roadmap.md",
        "references/harness-engineering.md",
    ]
    issues: list[HarnessIssue] = []
    for ref in required_refs:
        if ref not in text:
            issues.append(
                HarnessIssue(
                    code="docs_index_missing_ref",
                    message=f"docs index is missing reference to {ref}",
                    path="docs/index.md",
                )
            )
    return issues


def _check_plan_files(root: Path) -> list[HarnessIssue]:
    issues: list[HarnessIssue] = []
    plan_dirs = [
        root / "docs" / "exec-plans" / "active",
        root / "docs" / "exec-plans" / "completed",
    ]
    for plan_dir in plan_dirs:
        if not plan_dir.exists():
            issues.append(
                HarnessIssue(
                    code="missing_plan_directory",
                    message="plan directory is missing",
                    path=str(plan_dir.relative_to(root)),
                )
            )
            continue
        for plan_file in sorted(plan_dir.glob("*.md")):
            if plan_file.name == "README.md":
                continue
            text = plan_file.read_text(encoding="utf-8")
            for heading in PLAN_REQUIRED_HEADINGS:
                if heading not in text:
                    issues.append(
                        HarnessIssue(
                            code="plan_missing_heading",
                            message=f"missing heading {heading}",
                            path=str(plan_file.relative_to(root)),
                        )
                    )
    return issues


def _check_module_sizes(root: Path) -> list[HarnessIssue]:
    issues: list[HarnessIssue] = []
    for path in sorted((root / "src" / "calosum").rglob("*.py")):
        if path.name in {"__init__.py"}:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > MAX_MODULE_LINES:
            issues.append(
                HarnessIssue(
                    code="module_too_large",
                    message=f"module has {len(lines)} lines; keep under {MAX_MODULE_LINES}",
                    path=str(path.relative_to(root)),
                )
            )
    return issues


def _check_import_boundaries(root: Path) -> list[HarnessIssue]:
    issues: list[HarnessIssue] = []
    for path in sorted((root / "src" / "calosum").rglob("*.py")):
        if path.name == "__init__.py":
            if str(path.parent.name) != "calosum":
                # Only check main __init__.py boundaries
                continue

        rel_path = path.relative_to(root / "src" / "calosum")
        
        if str(rel_path.parent) == ".":
            module_name = path.stem
        else:
            module_name = f"{str(rel_path.parent).replace('/', '.')}.{path.stem}"

        allowed = MODULE_RULES.get(module_name)
        if allowed is None:
            issues.append(
                HarnessIssue(
                    code="missing_module_rule",
                    message="module is not registered in harness boundary rules",
                    path=str(path.relative_to(root)),
                )
            )
            continue
            
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _internal_imports(tree):
            if imported not in allowed:
                issues.append(
                    HarnessIssue(
                        code="forbidden_internal_import",
                        message=f"{module_name} must not import {imported}",
                        path=str(path.relative_to(root)),
                    )
                )
    return issues


def _internal_imports(tree: ast.AST) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("calosum."):
                parts = node.module.split(".")
                if len(parts) >= 3:
                    imports.add(f"{parts[1]}.{parts[2]}")
                elif len(parts) == 2:
                    imports.add(parts[1])
            elif node.level == 1 and node.module:
                imports.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("calosum."):
                    parts = alias.name.split(".")
                    if len(parts) >= 3:
                        imports.add(f"{parts[1]}.{parts[2]}")
                    elif len(parts) == 2:
                        imports.add(parts[1])
    return imports


if __name__ == "__main__":
    raise SystemExit(main())
