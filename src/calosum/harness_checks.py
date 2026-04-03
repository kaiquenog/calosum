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
    "shared/models",
    "shared/utils",
    "domain",
    "domain/agent",
    "domain/cognition",
    "domain/metacognition",
    "domain/memory",
    "domain/execution",
    "domain/infrastructure",
    "adapters",
    "adapters/execution",
    "adapters/infrastructure",
    "bootstrap",
    "bootstrap/entry",
    "bootstrap/wiring",
    "bootstrap/infrastructure",
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
    "shared.models.types": set(),
    "shared.models.jepa": set(),
    "shared.models.ports": {"shared.models.jepa", "shared.models.types", "domain.metacognition.metacognition"},
    "shared.models.schemas": set(),
    "shared.utils.async_utils": set(),
    "shared.utils.serialization": set(),
    "shared.utils.tools": {"shared.models.types"},

    # DOMAIN
    "domain.agent.agent_config": set(),
    "domain.execution.agent_execution": {"shared.utils.async_utils", "shared.models.ports", "shared.models.types", "domain.execution.execution_utils"},
    "domain.cognition.bridge": {"shared.models.types", "shared.models.ports"},
    "domain.infrastructure.event_bus": set(),
    "domain.cognition.differentiable_logic": {"shared.models.types"},
    "domain.cognition.right_hemisphere": {"shared.models.types"},
    "domain.cognition.left_hemisphere": {"shared.models.types", "domain.cognition.differentiable_logic"},
    "domain.execution.runtime": {"shared.models.types"},
    "domain.memory.memory": {"shared.models.types", "shared.models.ports"},
    "domain.memory.persistent_memory": {"domain.memory.memory", "shared.utils.serialization", "shared.models.types"},
    "domain.infrastructure.telemetry": {"shared.models.types", "shared.utils.serialization"},

    "domain.execution.workspace": {"domain.agent.orchestrator", "shared.models.types"},
    "domain.metacognition.introspection": {"shared.models.types"},
    "domain.metacognition.introspection_capabilities": {"domain.agent.orchestrator", "bootstrap.wiring.factory", "shared.models.types"},
    "domain.execution.execution_utils": {"shared.models.types"},
    "domain.agent.evolution": {"shared.models.types", "domain.agent.agent_config", "shared.models.ports", "domain.agent.directive_guardrails"},
    "domain.agent.directive_guardrails": set(),
    "domain.agent.idle_foraging": {"shared.models.types"},
    "domain.infrastructure.interceptors": {"domain.infrastructure.event_bus"},
    "domain.metacognition.metacognition": {"domain.cognition.bridge", "shared.models.types", "domain.cognition.differentiable_logic"},
    "domain.agent.multiagent": {"domain.infrastructure.event_bus", "shared.models.types"},
    "domain.metacognition.self_model": {"domain.agent.orchestrator", "shared.models.types"},
    "domain.infrastructure.verifier": {"shared.models.schemas", "shared.models.types"},
    "domain.metacognition.awareness": {"shared.models.types", "domain.metacognition.introspection", "domain.agent.orchestrator", "domain.agent.directive_guardrails", "shared.utils.async_utils"},
    "domain.execution.group_turn": {"shared.utils.async_utils", "shared.models.types", "domain.execution.workspace", "domain.metacognition.metacognition", "domain.metacognition.awareness", "domain.agent.orchestrator"},
    "domain.agent.orchestrator_briefing": set(),
    "domain.agent.orchestrator_utils": {"shared.models.types"},
    "domain.agent.orchestrator": {
        "shared.utils.async_utils",
        "domain.agent.agent_config",
        "domain.agent.orchestrator_briefing",
        "domain.agent.orchestrator_utils",
        "domain.metacognition.awareness",
        "domain.cognition.bridge",
        "domain.infrastructure.event_bus",
        "domain.agent.evolution",
        "domain.agent.directive_guardrails",
        "domain.execution.agent_execution",
        "domain.execution.group_turn",
        "domain.agent.idle_foraging",
        "domain.metacognition.introspection",
        "domain.cognition.left_hemisphere",
        "domain.memory.memory",
        "domain.metacognition.metacognition",
        "shared.models.ports",
        "domain.cognition.right_hemisphere",
        "domain.execution.runtime",
        "domain.metacognition.self_model",
        "domain.infrastructure.telemetry",
        "domain.infrastructure.verifier",
        "domain.execution.workspace",
        "shared.models.types",
    },

    # BOOTSTRAP
    "bootstrap.infrastructure.settings": {
        "bootstrap.infrastructure.helpers",
    },
    "bootstrap.infrastructure.helpers": {
        "bootstrap.infrastructure.settings",
    },
    "bootstrap.wiring.backend_resolvers": {
        "adapters.perception.active_inference",
        "adapters.bridge.bridge_cross_attention",
        "adapters.infrastructure.contract_wrappers",
        "adapters.experience.gea_experience_graph",
        "adapters.experience.gea_experience_store",
        "adapters.experience.gea_reflection_experience",
        "adapters.hemisphere.left_hemisphere_rlm",
        "adapters.llm.llm_failover",
        "adapters.llm.llm_fusion",
        "adapters.llm.llm_qwen",
        "adapters.perception.multimodal_perception",
        "adapters.hemisphere.right_hemisphere_hf",
        "adapters.hemisphere.right_hemisphere_heuristic_jepa",
        "adapters.hemisphere.right_hemisphere_trained_jepa",
        "adapters.hemisphere.right_hemisphere_jepars",
        "adapters.hemisphere.right_hemisphere_vjepa21",
        "adapters.hemisphere.right_hemisphere_vljepa",
        "bootstrap.infrastructure.settings",
        "domain.metacognition.metacognition",
        "domain.cognition.right_hemisphere",
    },
    "bootstrap.wiring.factory": {
        "adapters.perception.active_inference",
        "adapters.execution.action_runtime",
        "adapters.bridge.bridge_store",
        "bootstrap.wiring.agent_baseline",
        "bootstrap.wiring.backend_resolvers",
        "adapters.knowledge.knowledge_graph_nanorag",
        "adapters.communication.latent_exchange",
        "adapters.llm.llm_qwen",
        "adapters.tools.mcp_client",
        "adapters.memory.memory_qdrant",
        "adapters.night_trainer.night_trainer",
        "adapters.perception.quantized_embeddings",
        "adapters.communication.telemetry_otlp",
        "adapters.memory.text_embeddings",
        "adapters.hemisphere.right_hemisphere_heuristic_jepa",
        "adapters.hemisphere.right_hemisphere_trained_jepa",
        "domain.cognition.bridge",
        "domain.infrastructure.event_bus",
        "domain.agent.evolution",
        "domain.metacognition.introspection_capabilities",
        "domain.infrastructure.interceptors",
        "domain.memory.memory",
        "domain.agent.orchestrator",
        "domain.memory.persistent_memory",
        "domain.cognition.right_hemisphere",
        "bootstrap.infrastructure.settings",
        "domain.infrastructure.telemetry",
        "shared.models.ports",
        "shared.models.types"
    },
    "bootstrap.wiring.agent_baseline": {
        "adapters.execution.action_runtime",
        "adapters.llm.llm_qwen",
        "adapters.memory.text_embeddings",
        "bootstrap.infrastructure.settings",
        "shared.models.types",
    },
    "bootstrap.infrastructure.jepa_rs_manager": {"shared.models.types"},
    "bootstrap.entry.cli": {
        "bootstrap.wiring.factory",
        "domain.metacognition.metacognition",
        "shared.utils.serialization",
        "bootstrap.infrastructure.settings",
        "domain.infrastructure.telemetry",
        "shared.models.types",
    },
    "bootstrap.entry.api": {
        "adapters.communication.channel_telegram", "bootstrap.wiring.factory", "bootstrap.infrastructure.settings", "domain.metacognition.introspection",
        "shared.utils.serialization", "shared.models.types", "bootstrap.routers", "bootstrap.entry.context",
        "bootstrap.routers.system", "bootstrap.routers.chat", "bootstrap.routers.telemetry",
    },
    "bootstrap.entry.context": {"bootstrap.wiring.factory", "bootstrap.infrastructure.settings"},
    "bootstrap.routers.system": {"bootstrap.entry.context", "shared.utils.serialization"},
    "bootstrap.routers.telemetry": {"bootstrap.entry.context"},
    "bootstrap.routers.chat": {"bootstrap.entry.context", "shared.models.types", "shared.utils.serialization"},
    "bootstrap.entry.__main__": {"bootstrap.entry.cli"},

    # ADAPTERS
    "adapters.execution.action_runtime": {
        "adapters.tools.http_request", "adapters.tools.code_execution", "adapters.tools.introspection",
        "adapters.tools.mcp_tool", "adapters.tools.persistent_shell", "adapters.tools.subordinate_agent",
        "shared.utils.async_utils", "shared.utils.tools", "shared.models.types",
    },
    "adapters.bridge.bridge_cross_attention": set(),
    "adapters.bridge.bridge_store": {"shared.models.ports"},
    "adapters.communication.channel_telegram": {"shared.models.types"},
    "adapters.communication.latent_exchange": {"shared.models.ports", "domain.infrastructure.event_bus"},
    "adapters.communication.telemetry_otlp": {"domain.infrastructure.telemetry"},
    "adapters.infrastructure.contract_wrappers": {"shared.models.types"},
    "adapters.experience.gea_experience_distributed": set(),
    "adapters.experience.gea_experience_graph": {"shared.models.ports"},
    "adapters.experience.gea_experience_store": set(),
    "adapters.experience.gea_reflection_experience": {
        "adapters.experience.variant_preference",
        "domain.cognition.bridge",
        "domain.metacognition.metacognition",
    },
    "adapters.experience.variant_preference": set(),
    "adapters.hemisphere.left_hemisphere_rlm": {"shared.models.types", "shared.models.ports"},
    "adapters.hemisphere.right_hemisphere_heuristic_jepa": {"shared.models.jepa", "shared.models.types", "adapters.memory.text_embeddings"},
    "adapters.hemisphere.right_hemisphere_trained_jepa": {"shared.models.jepa", "shared.models.types"},
    "adapters.hemisphere.right_hemisphere_hf": {"shared.models.types", "domain.cognition.right_hemisphere", "shared.models.ports"},
    "adapters.hemisphere.right_hemisphere_jepars": {"shared.models.types"},
    "adapters.hemisphere.right_hemisphere_vjepa21": {"shared.models.types", "shared.models.ports"},
    "adapters.hemisphere.right_hemisphere_vljepa": {"adapters.hemisphere.right_hemisphere_vjepa21", "shared.models.types"},
    "adapters.knowledge.knowledge_graph_nanorag": {"shared.models.types"},
    "adapters.llm.llm_failover": {"shared.utils.async_utils", "shared.models.ports", "shared.models.types"},
    "adapters.llm.llm_fusion": {"shared.utils.async_utils", "shared.models.types"},
    "adapters.llm.llm_payload_parser": {"shared.models.types"},
    "adapters.llm.llm_payloads": {"shared.models.types"},
    "adapters.llm.llm_qwen": {"adapters.llm.llm_payloads", "adapters.llm.llm_payload_parser", "shared.utils.async_utils", "shared.models.types"},
    "adapters.memory.memory_qdrant": {
        "adapters.memory.text_embeddings",
        "adapters.memory.memory_qdrant_serializers",
        "shared.utils.async_utils",
        "shared.models.types",
        "domain.memory.memory",
        "shared.models.ports",
    },
    "adapters.memory.memory_qdrant_serializers": {"shared.models.types", "shared.models.ports"},
    "adapters.memory.text_embeddings": {"shared.utils.async_utils", "shared.models.ports"},
    "adapters.night_trainer.night_trainer": {
        "adapters.night_trainer.night_trainer_dspy",
        "adapters.night_trainer.night_trainer_lora",
    },
    "adapters.night_trainer.night_trainer_dspy": set(),
    "adapters.night_trainer.night_trainer_lora": set(),
    "adapters.perception.active_inference": {"shared.models.types", "domain.cognition.right_hemisphere"},
    "adapters.perception.multimodal_perception": {"shared.models.ports"},
    "adapters.perception.quantized_embeddings": {"shared.models.ports"},
    "adapters.tools.introspection": {"shared.models.types", "shared.utils.async_utils"},
    "adapters.tools.code_execution": {"shared.utils.tools"},
    "adapters.tools.http_request": {"shared.utils.tools"},
    "adapters.tools.mcp_client": set(),
    "adapters.tools.mcp_tool": {"shared.utils.tools"},
    "adapters.tools.persistent_shell": {"shared.utils.tools"},
    "adapters.tools.subordinate_agent": {"domain.infrastructure.event_bus", "domain.agent.multiagent", "shared.utils.tools"},

    # ROOT
    "harness_checks": set(),
    "__main__": {"bootstrap.entry.cli"},
    "__init__": {
        "adapters.knowledge.knowledge_graph_nanorag",
        "adapters.perception.active_inference",
        "adapters.bridge.bridge_cross_attention",
        "adapters.hemisphere.left_hemisphere_rlm",
        "adapters.perception.multimodal_perception",
        "adapters.hemisphere.right_hemisphere_heuristic_jepa",
        "adapters.hemisphere.right_hemisphere_trained_jepa",
        "adapters.hemisphere.right_hemisphere_jepars",
        "adapters.hemisphere.right_hemisphere_vjepa21",
        "adapters.hemisphere.right_hemisphere_vljepa",
        "bootstrap.wiring.factory",
        "bootstrap.wiring.agent_baseline",
        "domain.cognition.bridge",
        "domain.cognition.left_hemisphere",
        "domain.memory.memory",
        "domain.metacognition.metacognition",
        "domain.agent.multiagent",
        "domain.agent.orchestrator",
        "domain.memory.persistent_memory",
        "shared.models.ports",
        "shared.models.jepa",
        "domain.cognition.right_hemisphere",
        "domain.execution.runtime",
        "domain.metacognition.self_model",
        "domain.execution.workspace",
        "domain.metacognition.introspection",
        "domain.agent.evolution",
        "shared.utils.serialization",
        "bootstrap.infrastructure.settings",
        "domain.infrastructure.telemetry",
        "domain.infrastructure.verifier",
        "shared.models.types",
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
