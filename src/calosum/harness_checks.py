from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "calosum"
DOCS_ROOT = REPO_ROOT / "docs"

SEMANTIC_PACKAGES = [
    "shared",
    "domain",
    "adapters",
    "bootstrap",
    "bootstrap/routers",
]

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
    "shared.schemas": set(),
    "shared.async_utils": set(),
    "shared.serialization": set(),
    "shared.tools": {"shared.types"},

    # DOMAIN
    "domain.agent_config": set(),
    "domain.agent_execution": {"shared.async_utils", "shared.ports", "shared.types", "domain.execution_utils"},
    "domain.bridge": {"shared.types", "shared.ports"},
    "domain.event_bus": set(),
    "domain.differentiable_logic": {"shared.types"},
    "domain.right_hemisphere": {"shared.types"},
    "domain.left_hemisphere": {"shared.types", "domain.differentiable_logic"},
    "domain.runtime": {"domain.runtime_dsl", "shared.types"},
    "domain.runtime_dsl": {"shared.types"},
    "domain.memory": {"shared.types", "shared.ports"},
    "domain.persistent_memory": {"domain.memory", "shared.serialization", "shared.types"},
    "domain.telemetry": {"shared.types", "shared.serialization"},

    "domain.workspace": {"domain.orchestrator", "shared.types"},
    "domain.introspection": {"shared.types"},
    "domain.introspection_capabilities": {"domain.orchestrator", "bootstrap.factory", "shared.types"},
    "domain.execution_utils": {"shared.types"},
    "domain.evolution": {"shared.types", "domain.agent_config", "shared.ports", "domain.directive_guardrails"},
    "domain.directive_guardrails": set(),
    "domain.idle_foraging": {"shared.types"},
    "domain.interceptors": {"domain.event_bus"},
    "domain.metacognition": {"domain.bridge", "shared.types", "domain.differentiable_logic"},
    "domain.multiagent": {"domain.event_bus", "shared.types"},
    "domain.self_model": {"domain.orchestrator", "shared.types"},
    "domain.verifier": {"shared.schemas", "shared.types"},
    "domain.awareness": {"shared.types", "domain.introspection", "domain.orchestrator", "domain.directive_guardrails"},
    "domain.group_turn": {"shared.async_utils", "shared.types", "domain.workspace", "domain.metacognition", "domain.awareness", "domain.orchestrator"},
    "domain.orchestrator": {
        "shared.async_utils",
        "domain.agent_config",
        "domain.awareness",
        "domain.bridge",
        "domain.event_bus",
        "domain.evolution",
        "domain.directive_guardrails",
        "domain.agent_execution",
        "domain.group_turn",
        "domain.idle_foraging",
        "domain.introspection",
        "domain.left_hemisphere",
        "domain.memory",
        "domain.metacognition",
        "shared.ports",
        "domain.right_hemisphere",
        "domain.runtime",
        "domain.self_model",
        "domain.telemetry",
        "domain.verifier",
        "domain.workspace",
        "shared.types",
    },

    # BOOTSTRAP
    "bootstrap.settings": set(),
    "bootstrap.backend_resolvers": {
        "adapters.active_inference",
        "adapters.bridge_cross_attention",
        "adapters.contract_wrappers",
        "adapters.gea_experience_graph",
        "adapters.gea_experience_store",
        "adapters.gea_reflection_experience",
        "adapters.left_hemisphere_rlm",
        "adapters.llm_failover",
        "adapters.llm_qwen",
        "adapters.multimodal_perception",
        "adapters.right_hemisphere_hf",
        "adapters.right_hemisphere_jepars",
        "adapters.right_hemisphere_vjepa21",
        "adapters.right_hemisphere_vljepa",
        "bootstrap.settings",
        "domain.metacognition",
        "domain.right_hemisphere",
    },
    "bootstrap.factory": {
        "adapters.active_inference",
        "adapters.action_runtime",
        "adapters.bridge_store",
        "bootstrap.backend_resolvers",
        "adapters.knowledge_graph_nanorag",
        "adapters.latent_exchange",
        "adapters.llm_qwen",
        "adapters.mcp_client",
        "adapters.memory_qdrant",
        "adapters.night_trainer",
        "adapters.quantized_embeddings",
        "adapters.telemetry_otlp",
        "adapters.text_embeddings",
        "domain.bridge",
        "domain.event_bus",
        "domain.evolution",
        "domain.introspection_capabilities",
        "domain.interceptors",
        "domain.memory",
        "domain.orchestrator",
        "domain.persistent_memory",
        "domain.right_hemisphere",
        "bootstrap.settings",
        "domain.telemetry",
        "shared.ports",
        "shared.types"
    },
    "bootstrap.jepa_rs_manager": {"shared.types"},
    "bootstrap.cli": {
        "bootstrap.factory",
        "domain.metacognition",
        "shared.serialization",
        "bootstrap.settings",
        "domain.telemetry",
        "shared.types",
    },
    "bootstrap.api": {
        "adapters.channel_telegram", "bootstrap.factory", "bootstrap.settings", "domain.introspection",
        "shared.serialization", "shared.types", "bootstrap.routers", "bootstrap.context",
        "bootstrap.routers.system", "bootstrap.routers.chat", "bootstrap.routers.telemetry",
    },
    "bootstrap.context": {"bootstrap.factory", "bootstrap.settings"},
    "bootstrap.routers.system": {"bootstrap.context", "shared.serialization"},
    "bootstrap.routers.telemetry": {"bootstrap.context"},
    "bootstrap.routers.chat": {"bootstrap.context", "shared.types", "shared.serialization"},
    "bootstrap.__main__": {"bootstrap.cli"},

    # ADAPTERS
    "adapters.active_inference": {"shared.types", "domain.right_hemisphere"},
    "adapters.bridge_cross_attention": set(),
    "adapters.contract_wrappers": {"shared.types"},
    "adapters.action_runtime": {
        "adapters.tools.http_request", "adapters.tools.code_execution", "adapters.tools.introspection",
        "adapters.tools.mcp_tool", "adapters.tools.persistent_shell", "adapters.tools.subordinate_agent",
        "shared.async_utils", "shared.tools", "shared.types",
    },
    "adapters.bridge_store": {"shared.ports"},
    "adapters.channel_telegram": {"shared.types"},
    "adapters.knowledge_graph_nanorag": {"shared.types"},
    "adapters.mcp_client": set(),
    "adapters.gea_experience_graph": {"shared.ports"},
    "adapters.gea_experience_store": set(),
    "adapters.gea_experience_distributed": set(),
    "adapters.gea_reflection_experience": {"domain.bridge", "domain.metacognition"},
    "adapters.left_hemisphere_rlm": {"shared.types", "shared.ports"},
    "adapters.llm_failover": {"shared.async_utils", "shared.ports", "shared.types"},
    "adapters.llm_payload_parser": {"shared.types"},
    "adapters.llm_payloads": {"shared.types"},
    "adapters.llm_qwen": {"adapters.llm_payloads", "adapters.llm_payload_parser", "shared.async_utils", "shared.types"},
    "adapters.memory_qdrant": {
        "adapters.text_embeddings",
        "adapters.memory_qdrant_serializers",
        "shared.async_utils",
        "shared.types",
        "domain.memory",
        "shared.ports",
    },
    "adapters.memory_qdrant_serializers": {"shared.types", "shared.ports"},
    "adapters.quantized_embeddings": {"shared.ports"},
    "adapters.right_hemisphere_hf": {"shared.types", "domain.right_hemisphere", "shared.ports"},
    "adapters.right_hemisphere_jepars": {"shared.types"},
    "adapters.right_hemisphere_vjepa21": {"shared.types", "shared.ports"},
    "adapters.right_hemisphere_vljepa": {"adapters.right_hemisphere_vjepa21", "shared.types"},
    "adapters.telemetry_otlp": {"domain.telemetry"},
    "adapters.multimodal_perception": {"shared.ports"},
    "adapters.latent_exchange": {"shared.ports", "domain.event_bus"},
    "adapters.text_embeddings": {"shared.async_utils", "shared.ports"},
    "adapters.night_trainer": {"adapters.night_trainer_dspy"},
    "adapters.night_trainer_dspy": set(),
    "adapters.night_trainer_lora": set(),
    "adapters.tools.introspection": {"shared.types"},
    "adapters.tools.code_execution": {"shared.tools"},
    "adapters.tools.http_request": {"shared.tools"},
    "adapters.tools.mcp_tool": {"shared.tools"},
    "adapters.tools.persistent_shell": {"shared.tools"},
    "adapters.tools.subordinate_agent": {"domain.event_bus", "domain.multiagent", "shared.tools"},

    # ROOT
    "harness_checks": set(),
    "__init__": {
        "adapters.active_inference",
        "adapters.bridge_cross_attention",
        "adapters.knowledge_graph_nanorag",
        "adapters.left_hemisphere_rlm",
        "adapters.multimodal_perception",
        "adapters.right_hemisphere_jepars",
        "adapters.right_hemisphere_vjepa21",
        "adapters.right_hemisphere_vljepa",
        "bootstrap.factory",
        "domain.bridge",
        "domain.left_hemisphere",
        "domain.memory",
        "domain.metacognition",
        "domain.multiagent",
        "domain.orchestrator",
        "domain.persistent_memory",
        "shared.ports",
        "domain.right_hemisphere",
        "domain.runtime",
        "domain.self_model",
        "domain.workspace",
        "domain.introspection",
        "domain.evolution",
        "shared.serialization",
        "bootstrap.settings",
        "domain.telemetry",
        "domain.verifier",
        "shared.types",
    },
}

MAX_MODULE_LINES = 400


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
    issues.extend(_check_package_docstrings(root))
    issues.extend(_check_shared_domain_runtime_imports(root))
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
    return [HarnessIssue("missing_required_path", "required harness artifact is missing", str(p.relative_to(REPO_ROOT)))
            for p in REQUIRED_PATHS if not (root / p.relative_to(REPO_ROOT)).exists()]

def _check_agents_map(root: Path) -> list[HarnessIssue]:
    path = root / "AGENTS.md"
    if not path.exists(): return [HarnessIssue("missing_agents_map", "AGENTS.md map is missing from root")]
    text = path.read_text(encoding="utf-8")
    issues = []
    if len(text.splitlines()) > 120: issues.append(HarnessIssue("agents_too_long", "AGENTS.md should remain a short map", "AGENTS.md"))
    for link in REQUIRED_DOC_LINKS:
        if link not in text: issues.append(HarnessIssue("agents_missing_link", f"missing link to {link}", "AGENTS.md"))
    return issues

def _check_docs_index(root: Path) -> list[HarnessIssue]:
    path = root / "docs" / "index.md"
    if not path.exists(): return [HarnessIssue("missing_docs_index", "docs/index.md is missing")]
    text = path.read_text(encoding="utf-8")
    required = ["ARCHITECTURE.md", "PLANS.md", "QUALITY_SCORE.md", "RELIABILITY.md", "INFRASTRUCTURE.md", "production-roadmap.md", "references/harness-engineering.md"]
    return [HarnessIssue("docs_index_missing_ref", f"missing reference to {ref}", "docs/index.md") for ref in required if ref not in text]

def _check_plan_files(root: Path) -> list[HarnessIssue]:
    issues = []
    plan_dirs = [root / "docs" / "exec-plans" / "active", root / "docs" / "exec-plans" / "completed"]
    for d in plan_dirs:
        if not d.exists(): issues.append(HarnessIssue("missing_plan_directory", "plan directory is missing", str(d.relative_to(root)))); continue
        for f in sorted(d.glob("*.md")):
            if f.name == "README.md": continue
            text = f.read_text(encoding="utf-8")
            for h in PLAN_REQUIRED_HEADINGS:
                if h not in text: issues.append(HarnessIssue("plan_missing_heading", f"missing heading {h}", str(f.relative_to(root))))
    return issues

def _check_module_sizes(root: Path) -> list[HarnessIssue]:
    issues = []
    for p in sorted((root / "src" / "calosum").rglob("*.py")):
        if p.name == "__init__.py": continue
        if p.name == "harness_checks.py": continue  # governance tool, exempt from size limit
        lines = len(p.read_text(encoding="utf-8").splitlines())
        if lines > MAX_MODULE_LINES:
            issues.append(HarnessIssue("module_too_large", f"module has {lines} lines (max {MAX_MODULE_LINES})", str(p.relative_to(root))))
    return issues


def _check_import_boundaries(root: Path) -> list[HarnessIssue]:
    issues = []
    for p in sorted((root / "src" / "calosum").rglob("*.py")):
        if p.name == "__init__.py" and str(p.parent.name) != "calosum": continue
        rel = p.relative_to(root / "src" / "calosum")
        mod = p.stem if str(rel.parent) == "." else f"{str(rel.parent).replace('/', '.')}.{p.stem}"
        allowed = MODULE_RULES.get(mod)
        if allowed is None:
            issues.append(HarnessIssue("missing_module_rule", f"module {mod} not registered", str(p.relative_to(root))))
            continue
        for imp in _internal_imports(ast.parse(p.read_text(encoding="utf-8"))):
            if imp not in allowed:
                issues.append(HarnessIssue("forbidden_internal_import", f"{mod} forbidden import: {imp}", str(p.relative_to(root))))
    return issues

def _internal_imports(tree: ast.AST) -> set[str]:
    imps = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("calosum."):
                p = node.module.split(".")
                if len(p) >= 4: imps.add(f"{p[1]}.{p[2]}.{p[3]}")
                elif len(p) == 3: imps.add(f"{p[1]}.{p[2]}")
                else: imps.add(p[1])
            elif node.level == 1: imps.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("calosum."):
                    p = a.name.split(".")
                    if len(p) >= 4: imps.add(f"{p[1]}.{p[2]}.{p[3]}")
                    elif len(p) == 3: imps.add(f"{p[1]}.{p[2]}")
                    else: imps.add(p[1])
    return imps

def _check_package_docstrings(root: Path) -> list[HarnessIssue]:
    """Require every semantic package __init__.py to have a module-level docstring."""
    issues = []
    for pkg in SEMANTIC_PACKAGES:
        init = root / "src" / "calosum" / pkg / "__init__.py"
        if not init.exists():
            continue
        text = init.read_text(encoding="utf-8").strip()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        has_docstring = (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        )
        if not has_docstring:
            issues.append(HarnessIssue(
                "missing_package_docstring",
                f"package {pkg}/__init__.py is missing a module-level docstring",
                str(init.relative_to(root)),
            ))
    return issues


def _check_shared_domain_runtime_imports(root: Path) -> list[HarnessIssue]:
    """Ensure shared/ modules never import from domain.* at runtime (TYPE_CHECKING-only is fine)."""
    issues = []
    shared_dir = root / "src" / "calosum" / "shared"
    for p in sorted(shared_dir.rglob("*.py")):
        try:
            source = p.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, OSError):
            continue

        # Collect all node IDs that live inside `if TYPE_CHECKING:` blocks
        type_checking_nodes: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test = node.test
                is_tc = (
                    (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
                    or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")
                )
                if is_tc:
                    for child in ast.walk(node):
                        type_checking_nodes.add(id(child))

        for node in ast.walk(tree):
            if id(node) in type_checking_nodes:
                continue
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("calosum.domain"):
                    issues.append(HarnessIssue(
                        "shared_domain_runtime_import",
                        f"shared module imports from domain at runtime: {node.module}",
                        str(p.relative_to(root)),
                    ))
    return issues


if __name__ == "__main__":
    raise SystemExit(main())
