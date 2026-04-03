from __future__ import annotations

from dataclasses import dataclass

from calosum.shared.models.types import (
    ActionExecutionResult,
    CognitiveBridgePacket,
    LeftHemisphereResult,
    MemoryContext,
    PrimitiveAction,
    TypedLambdaProgram,
    UserTurn,
    CognitiveWorkspace,
)


@dataclass(slots=True)
class LeftHemisphereLogicalSLMConfig:
    model_name: str = "mistral-small-mx-placeholder"
    lambda_runtime: str = "lambda_recursive_runtime_v0"
    max_actions: int = 3


class LeftHemisphereLogicalSLM:
    """
    Hemisferio esquerdo orientado a linguagem, logica e execucao.

    Em producao, o modelo seria um SLM quantizado em MX e especializado com
    distilacao agentica. O contrato importante aqui e:
    - recebe soft prompts afetivos e sinais de controle;
    - consulta memoria dual;
    - devolve plano simbolico e acoes primitivas tipificadas.
    """

    def __init__(self, config: LeftHemisphereLogicalSLMConfig | None = None) -> None:
        self.config = config or LeftHemisphereLogicalSLMConfig()

    def reason(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
        workspace: CognitiveWorkspace | None = None,
    ) -> LeftHemisphereResult:
        # Ground reasoning in differentiable logic (V2)
        from calosum.domain.cognition.differentiable_logic import LogicTensorNetwork
        ltn = LogicTensorNetwork()
        grounding_scores = [
            ltn.ground_rule(rule.statement, bridge_packet.latent_vector)
            for rule in memory_context.semantic_rules
        ]
        avg_grounding = sum(grounding_scores) / max(1, len(grounding_scores))

        rules = [rule.statement for rule in memory_context.semantic_rules[:2]]
        knowledge_facts = memory_context.knowledge_triples[:3]
        empathy_priority = bridge_packet.control.empathy_priority
        short_response = self._prefers_short_response(memory_context)
        stepwise_structure = self._prefers_stepwise_structure(memory_context)
        wants_plan = self._wants_plan(user_turn)
        plan_steps = self._build_plan_steps(user_turn, empathy_priority) if wants_plan else []
        response_text = self._compose_response_text(
            empathy_priority=empathy_priority,
            short_response=short_response,
            wants_plan=wants_plan,
            plan_steps=plan_steps,
        )

        # DSPy Integration: Inject optimized prompts from persistent evolution
        optimized_system_prompt = self._load_optimized_prompt()
        if optimized_system_prompt:
            bridge_packet.control.system_directives.insert(0, f"[DSPy Cognitive Model]: {optimized_system_prompt}")

        actions = [
            PrimitiveAction(
                action_type="respond_text",
                typed_signature="ResponsePlan -> SafeTextMessage",
                payload={
                    "text": response_text,
                    "temperature": bridge_packet.control.target_temperature,
                    "soft_prompts": [token.token for token in bridge_packet.soft_prompts],
                    "grounding_confidence": avg_grounding,
                },
                safety_invariants=[
                    "never execute external side effects without explicit tool approval",
                    "keep reasoning aligned with typed runtime constraints",
                ],
            )
        ]

        if wants_plan and len(actions) < self.config.max_actions:
            actions.append(
                PrimitiveAction(
                    action_type="propose_plan",
                    typed_signature="DecisionContext -> TypedPlan",
                    payload={
                        "steps": plan_steps,
                        "style": "short" if short_response else "detailed",
                        "stepwise": stepwise_structure or wants_plan,
                    },
                    safety_invariants=[
                        "plan must remain advisory until explicitly executed",
                        "steps must stay inside validated action vocabulary",
                    ],
                )
            )

        if rules and len(actions) < self.config.max_actions:
            actions.append(
                PrimitiveAction(
                    action_type="load_semantic_rules",
                    typed_signature="MemoryContext -> RuleSet",
                    payload={"rules": rules},
                    safety_invariants=["only reference consolidated rules with provenance"],
                )
            )

        lambda_program = TypedLambdaProgram(
            signature="Context -> Memory -> Decision",
            expression=self._build_lambda_expression(actions),
            expected_effect="Generate an empathetic and logically constrained decision.",
        )

        reasoning_summary = [
            f"bridge_salience={bridge_packet.salience}",
            f"empathy_priority={empathy_priority}",
            f"episodic_memories={len(memory_context.recent_episodes)}",
            f"semantic_rules={len(memory_context.semantic_rules)}",
            f"knowledge_triples={len(memory_context.knowledge_triples)}",
            f"wants_plan={wants_plan}",
            f"response_style={'short' if short_response else 'standard'}",
            f"runtime_feedback_items={len(runtime_feedback or [])}",
            f"attempt={attempt}",
        ]

        result = LeftHemisphereResult(
            response_text=response_text,
            lambda_program=lambda_program,
            actions=actions,
            reasoning_summary=reasoning_summary,
            telemetry={
                "model_name": self.config.model_name,
                "lambda_runtime": self.config.lambda_runtime,
                "system_directives": bridge_packet.control.system_directives,
                "response_style": "short" if short_response else "standard",
                "plan_steps": plan_steps,
                "knowledge_facts": [
                    f"{fact.subject}:{fact.predicate}:{fact.object}" for fact in knowledge_facts
                ],
                "runtime_feedback": runtime_feedback or [],
            },
        )

        if workspace is not None:
            workspace.left_notes.update({
                "response_text": result.response_text,
                "reasoning_summary": result.reasoning_summary,
                "actions": [a.action_type for a in result.actions],
            })

        return result

    def _build_lambda_expression(self, actions: list[PrimitiveAction]) -> str:
        """
        Gera o plano de execucao estruturado (JSON) em vez de DSL LISP.
        O plano descreve a sequencia de acoes a serem executadas.
        """
        import json
        if not actions:
            return json.dumps({"plan": []})

        # Em um cenário de Function Calling real, o LLM geraria este JSON.
        # Aqui, como estamos no domínio do orquestrador simulando o SLM,
        # mantemos a ordem das ações criadas.
        plan = [action.action_type for action in actions]
        return json.dumps({"plan": plan, "version": "2.0-structured"})

    async def areason(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
        workspace: CognitiveWorkspace | None = None,
    ) -> LeftHemisphereResult:
        return self.reason(
            user_turn=user_turn,
            bridge_packet=bridge_packet,
            memory_context=memory_context,
            runtime_feedback=runtime_feedback,
            attempt=attempt,
            workspace=workspace,
        )

    def repair(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        previous_result: LeftHemisphereResult,
        rejected_results: list[ActionExecutionResult],
        attempt: int,
        critique_feedback: list[str] | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> LeftHemisphereResult:
        feedback = []
        if critique_feedback:
            feedback.extend(critique_feedback)
        else:
            feedback.extend([
                f"{item.action_type}:{'; '.join(item.violations)}" for item in rejected_results
            ])
        return self.reason(
            user_turn=user_turn,
            bridge_packet=bridge_packet,
            memory_context=memory_context,
            runtime_feedback=feedback,
            attempt=attempt,
            workspace=workspace,
        )

    async def arepair(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        previous_result: LeftHemisphereResult,
        rejected_results: list[ActionExecutionResult],
        attempt: int,
        critique_feedback: list[str] | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> LeftHemisphereResult:
        return self.repair(
            user_turn=user_turn,
            bridge_packet=bridge_packet,
            memory_context=memory_context,
            previous_result=previous_result,
            rejected_results=rejected_results,
            attempt=attempt,
            critique_feedback=critique_feedback,
            workspace=workspace,
        )

    def _prefers_short_response(self, memory_context: MemoryContext) -> bool:
        joined_rules = " ".join(rule.statement.lower() for rule in memory_context.semantic_rules)
        if "respostas curtas" in joined_rules or "resposta curta" in joined_rules:
            return True
        return any(
            triple.predicate == "prefers_response_style" and triple.object == "short"
            for triple in memory_context.knowledge_triples
        )

    def _load_optimized_prompt(self) -> str | None:
        """Carrega o prompt otimizado gerado pelo DSPy na noite anterior, se existir."""
        import json
        import os
        from pathlib import Path
        
        # Em produção real isso viria de um DB ou Redis. Por ora, lemos do FS.
        artifact_path = Path(".calosum-runtime/dspy_artifacts/latest/compiled_prompt.json")
        if artifact_path.exists():
            try:
                data = json.loads(artifact_path.read_text(encoding="utf-8"))
                return data.get("selected_prompt")
            except Exception:
                pass
        return None

    def _prefers_stepwise_structure(self, memory_context: MemoryContext) -> bool:
        joined_rules = " ".join(rule.statement.lower() for rule in memory_context.semantic_rules)
        if "passos claros" in joined_rules or "plano" in joined_rules:
            return True
        return any(
            triple.predicate == "prefers_structure" and triple.object == "stepwise"
            for triple in memory_context.knowledge_triples
        )

    def _wants_plan(self, user_turn: UserTurn) -> bool:
        lowered = user_turn.user_text.lower()
        markers = ("plano", "passos", "organizar", "reorganizar", "roteiro")
        return any(marker in lowered for marker in markers)

    def _build_plan_steps(self, user_turn: UserTurn, empathy_priority: bool) -> list[str]:
        steps = [
            "Estabilizar o objetivo imediato e confirmar a restricao principal.",
            "Dividir o problema em subtarefas tipificadas e verificaveis.",
            "Executar apenas a proxima acao segura com validacao de resultado.",
        ]
        if empathy_priority:
            steps[0] = "Acolher o contexto emocional e definir o objetivo imediato mais seguro."
        if "projeto" in user_turn.user_text.lower():
            steps[1] = "Mapear o projeto em blocos menores com prioridade e criterio de parada."
        return steps

    def _compose_response_text(
        self,
        *,
        empathy_priority: bool,
        short_response: bool,
        wants_plan: bool,
        plan_steps: list[str],
    ) -> str:
        opening = (
            "Percebo alta carga emocional no contexto. "
            if empathy_priority
            else "Analisei o contexto e a memoria recente. "
        )
        if wants_plan and short_response:
            joined_steps = " ".join(
                f"{index + 1}. {step}" for index, step in enumerate(plan_steps[:3])
            )
            return opening + f"Plano seguro: {joined_steps}"
        if wants_plan:
            joined_steps = " ".join(
                f"Passo {index + 1}: {step}" for index, step in enumerate(plan_steps[:3])
            )
            return (
                opening
                + "Vou responder com um plano seguro, tipificado e alinhado ao estado atual. "
                + joined_steps
            )
        if short_response:
            return opening + "Resposta curta, segura e objetiva preparada."
        return opening + "Vou responder com um plano seguro, tipificado e alinhado ao estado atual."
