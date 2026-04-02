from __future__ import annotations

import json
import uuid
import logging
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from calosum.shared.models.ports import ActionRuntimePort, RightHemispherePort
    from calosum.domain.agent.agent_config import CalosumAgentConfig

from calosum.shared.models.types import DirectiveType, EvolutionDirective, SessionDiagnostic, utc_now

logger = logging.getLogger(__name__)


class EvolutionManager:
    """
    Gerencia o ciclo de vida de diretivas e diagnósticos de evolução.
    Encapsula aplicação de mudanças e persistência.
    """

    def __init__(
        self,
        archive: JsonlEvolutionArchive | None = None,
        proposer: EvolutionProposer | None = None,
    ) -> None:
        self.archive = archive
        self.proposer = proposer or EvolutionProposer()
        self.pending_directives: list[EvolutionDirective] = (
            self.archive.load_pending_directives() if self.archive else []
        )
        self.approved_prompt_directives: list[str] = (
            self.archive.load_applied_prompt_directives() if self.archive else []
        )

    def queue_directive(self, directive: EvolutionDirective) -> None:
        fingerprint = self._fingerprint(directive)
        existing = {
            self._fingerprint(item)
            for item in self.pending_directives
            if item.status == "pending"
        }
        if fingerprint in existing:
            return
        directive.status = "pending"
        self.pending_directives.append(directive)
        if self.archive:
            self.archive.record_directive(directive, event="queued")

    def apply_pending_directive(
        self, directive_id: str, context: Any
    ) -> EvolutionDirective | None:
        directive = next(
            (item for item in self.pending_directives if item.directive_id == directive_id),
            None,
        )
        if directive is None:
            return None
        self.pending_directives = [
            item for item in self.pending_directives if item.directive_id != directive_id
        ]
        self.apply_directive(directive, context)
        if self.archive:
            self.archive.record_directive(directive, event="manual_apply")
        return directive

    def apply_directive(self, directive: EvolutionDirective, context: Any) -> None:
        """
        Aplica as mudanças descritas na diretiva ao contexto do agente.
        """
        from calosum.domain.agent.directive_guardrails import (
            apply_controlled_right_hemisphere_params,
            apply_runtime_contract_audit_directive,
        )

        target, changes = directive.target_component, directive.proposed_change
        try:
            if directive.directive_type in {DirectiveType.TOPOLOGY, DirectiveType.ARCHITECTURE}:
                if apply_runtime_contract_audit_directive(context.action_runtime, directive):
                    return
                directive.status = "rejected_guardrail_topology_locked"
                return

            if directive.directive_type == DirectiveType.PARAMETER and target == "orchestrator":
                for k, v in changes.items():
                    if hasattr(context.config, k):
                        setattr(context.config, k, v)
                directive.status = "applied"
            elif (
                directive.directive_type == DirectiveType.PARAMETER
                and target == "orchestrator.branching_budget"
            ):
                for k, v in changes.items():
                    if hasattr(context.config.branching_budget, k):
                        setattr(context.config.branching_budget, k, v)
                directive.status = "applied"
            elif directive.directive_type == DirectiveType.PARAMETER and target == "bridge":
                if hasattr(context.tokenizer, "config"):
                    for k, v in changes.items():
                        if hasattr(context.tokenizer.config, k):
                            setattr(context.tokenizer.config, k, v)
                    directive.status = "applied"
            elif (
                directive.directive_type == DirectiveType.PARAMETER
                and target == "right_hemisphere"
            ):
                applied, rejected = apply_controlled_right_hemisphere_params(
                    context.right_hemisphere, changes
                )
                directive.proposed_change = {
                    **changes,
                    "_audit": {"applied": applied, "rejected": rejected},
                }
                directive.status = "applied" if applied else "rejected_guardrail_no_safe_change"
            elif directive.directive_type == DirectiveType.PROMPT and target in {
                "left_hemisphere",
                "reflection_controller",
            }:
                instruction = str(changes.get("instruction", "")).strip()
                if instruction and instruction not in self.approved_prompt_directives:
                    self.approved_prompt_directives.append(instruction)
                directive.status = "applied"
            else:
                directive.status = "rejected_guardrail_target_unknown"
        except Exception as e:
            logger.error(f"Failed to apply directive {directive.directive_id}: {e}")
            directive.status = "failed"

    def _fingerprint(self, directive: EvolutionDirective) -> str:
        return json.dumps(
            {
                "directive_type": directive.directive_type.value,
                "target_component": directive.target_component,
                "proposed_change": directive.proposed_change,
            },
            sort_keys=True,
        )


class EvolutionProposer:
    """
    Gera diretivas de evolução a partir de gargalos operacionais reais.
    """

    def propose(self, diagnostic: SessionDiagnostic) -> list[EvolutionDirective]:
        directives: list[EvolutionDirective] = []

        if diagnostic.tool_success_rate < 0.7:
            directives.append(
                EvolutionDirective(
                    directive_id=str(uuid.uuid4()),
                    directive_type=DirectiveType.PARAMETER,
                    target_component="orchestrator",
                    proposed_change={
                        "max_runtime_retries": min(4, max(2, round(diagnostic.average_retries) + 1)),
                    },
                    reasoning=(
                        f"Tool success rate caiu para {diagnostic.tool_success_rate:.1%}. "
                        "Aumentar retries reduz falhas transitórias enquanto os contratos são corrigidos."
                    ),
                )
            )

        if diagnostic.pending_approval_backlog > 0:
            directives.append(
                EvolutionDirective(
                    directive_id=str(uuid.uuid4()),
                    directive_type=DirectiveType.PROMPT,
                    target_component="left_hemisphere",
                    proposed_change={
                        "instruction": "prioritize clarification before side effects when approval backlog grows",
                        "pending_approval_backlog": diagnostic.pending_approval_backlog,
                    },
                    reasoning=(
                        f"Há {diagnostic.pending_approval_backlog} itens aguardando aprovação. "
                        "O runtime deve preferir perguntas de clarificação e caminhos sem side effects."
                    ),
                )
            )

        if diagnostic.dominant_variant == "default" and diagnostic.dominant_variant_ratio > 0.9:
            directives.append(
                EvolutionDirective(
                    directive_id=str(uuid.uuid4()),
                    directive_type=DirectiveType.PROMPT,
                    target_component="reflection_controller",
                    proposed_change={
                        "instruction": "bias selection toward non-default variants when diversity collapses",
                        "dominant_variant_ratio": diagnostic.dominant_variant_ratio,
                    },
                    reasoning=(
                        f"A variante default domina {diagnostic.dominant_variant_ratio:.1%} dos turnos recentes. "
                        "Isso justifica uma intervenção não paramétrica na política de reflexão."
                    ),
                )
            )

        if diagnostic.surprise_trend > 0.12:
            directives.append(
                EvolutionDirective(
                    directive_id=str(uuid.uuid4()),
                    directive_type=DirectiveType.PARAMETER,
                    target_component="right_hemisphere",
                    proposed_change={
                        "salience_smoothing_alpha": 0.35,
                        "salience_max_step": 0.16,
                    },
                    reasoning=(
                        f"A surpresa está em tendência de alta ({diagnostic.surprise_trend:+.3f}). "
                        "Aplicar amortecimento controlado de salience no hemisfério direito reduz reatividade sem trocar topologia."
                    ),
                )
            )

        # V2 Bayesian Optimization Hook: Tuning Active Inference Thresholds
        if diagnostic.average_surprise > 0.6 and diagnostic.tool_success_rate < 0.8:
            # Evidence of 'Under-branching': High surprise but low success 
            # implies we should branch more often to find better policies.
            directives.append(
                EvolutionDirective(
                    directive_id=str(uuid.uuid4()),
                    directive_type=DirectiveType.PARAMETER,
                    target_component="orchestrator",
                    proposed_change={
                        "surprise_threshold": max(0.3, diagnostic.average_surprise - 0.15),
                    },
                    reasoning="High average surprise coupled with low success indicates insufficient metacognitive intervention. Lowering surprise_threshold to trigger GEA more frequently.",
                )
            )

        return directives


class JsonlEvolutionArchive:
    """
    Persistência mínima para awareness e diretivas.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record_diagnostic(self, diagnostic: SessionDiagnostic) -> None:
        self._append({"record_type": "diagnostic", "recorded_at": utc_now().isoformat(),
                      "session_id": diagnostic.session_id, "diagnostic": _to_json_value(diagnostic)})

    def record_directive(self, directive: EvolutionDirective, *, event: str) -> None:
        self._append({"record_type": "directive", "recorded_at": utc_now().isoformat(),
                      "event": event, "directive": _to_json_value(directive)})

    def record_tool_trace(
        self, *, session_id: str, turn_id: str, tool_name: str,
        status: str, duration_ms: float, metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record individual tool execution trace for evolutionary analysis."""
        self._append({
            "record_type": "tool_trace",
            "recorded_at": utc_now().isoformat(),
            "session_id": session_id, "turn_id": turn_id,
            "tool_name": tool_name, "status": status,
            "duration_ms": round(duration_ms, 2),
            "metadata": metadata or {},
        })

    def record_execution_trace(
        self, *, session_id: str, turn_id: str,
        actions_proposed: int, actions_accepted: int,
        runtime_retries: int, critique_revisions: int,
        variant_id: str | None = None,
    ) -> None:
        """Record end-of-turn execution summary for GEA evolutionary history."""
        self._append({
            "record_type": "execution_trace",
            "recorded_at": utc_now().isoformat(),
            "session_id": session_id, "turn_id": turn_id,
            "variant_id": variant_id or "default",
            "actions_proposed": actions_proposed,
            "actions_accepted": actions_accepted,
            "runtime_retries": runtime_retries,
            "critique_revisions": critique_revisions,
        })

    def load_pending_directives(self) -> list[EvolutionDirective]:
        if self.path is None or not self.path.exists():
            return []

        latest_by_id: dict[str, EvolutionDirective] = {}
        for payload in self._read_records():
            if payload.get("record_type") != "directive":
                continue
            directive_raw = payload.get("directive")
            if not isinstance(directive_raw, dict):
                continue
            directive = _directive_from_dict(directive_raw)
            latest_by_id[directive.directive_id] = directive

        return [
            directive
            for directive in latest_by_id.values()
            if directive.status == "pending"
        ]

    def load_applied_prompt_directives(self) -> list[str]:
        if self.path is None or not self.path.exists():
            return []

        instructions: list[str] = []
        latest_by_id: dict[str, EvolutionDirective] = {}
        for payload in self._read_records():
            if payload.get("record_type") != "directive":
                continue
            directive_raw = payload.get("directive")
            if not isinstance(directive_raw, dict):
                continue
            directive = _directive_from_dict(directive_raw)
            latest_by_id[directive.directive_id] = directive

        for directive in latest_by_id.values():
            if directive.directive_type != DirectiveType.PROMPT or directive.status != "applied":
                continue
            instruction = str(directive.proposed_change.get("instruction", "")).strip()
            if instruction and instruction not in instructions:
                instructions.append(instruction)
        return instructions

    def _append(self, payload: dict[str, Any]) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def _read_records(self) -> list[dict[str, Any]]:
        if self.path is None or not self.path.exists():
            return []

        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
        return records


def _directive_from_dict(data: dict[str, Any]) -> EvolutionDirective:
    return EvolutionDirective(
        directive_id=str(data.get("directive_id", "")),
        directive_type=DirectiveType(str(data.get("directive_type", DirectiveType.PARAMETER.value))),
        target_component=str(data.get("target_component", "")),
        proposed_change=dict(data.get("proposed_change", {})),
        reasoning=str(data.get("reasoning", "")),
        status=str(data.get("status", "pending")),
        created_at=_datetime_from_value(data.get("created_at")),
    )


def _datetime_from_value(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return utc_now()


def _to_json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _to_json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    return value
