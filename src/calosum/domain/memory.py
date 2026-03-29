from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol
from uuid import uuid4

from calosum.shared.types import (
    ConsolidationReport,
    KnowledgeTriple,
    MemoryContext,
    MemoryEpisode,
    SemanticRule,
    UserTurn,
    utc_now,
)


class EpisodicMemoryStore(Protocol):
    def add(self, episode: MemoryEpisode) -> None: ...
    def query(self, user_turn: UserTurn, limit: int = 5) -> list[MemoryEpisode]: ...
    def all(self) -> list[MemoryEpisode]: ...


class SemanticMemoryStore(Protocol):
    def upsert(self, rule: SemanticRule) -> None: ...
    def all(self) -> list[SemanticRule]: ...


class SemanticGraphStore(Protocol):
    def upsert(self, triple: KnowledgeTriple) -> None: ...
    def all(self) -> list[KnowledgeTriple]: ...
    def query(self, user_turn: UserTurn, limit: int = 8) -> list[KnowledgeTriple]: ...


@dataclass(slots=True)
class InMemoryEpisodicStore:
    episodes: list[MemoryEpisode] = field(default_factory=list)

    def add(self, episode: MemoryEpisode) -> None:
        self.episodes.append(episode)

    def query(self, user_turn: UserTurn, limit: int = 5) -> list[MemoryEpisode]:
        query_terms = set(user_turn.user_text.lower().split())
        ranked = sorted(
            self.episodes,
            key=lambda episode: (
                len(query_terms.intersection(episode.user_turn.user_text.lower().split())),
                episode.recorded_at,
            ),
            reverse=True,
        )
        return ranked[:limit]

    def all(self) -> list[MemoryEpisode]:
        return list(self.episodes)


@dataclass(slots=True)
class InMemorySemanticStore:
    rules: dict[str, SemanticRule] = field(default_factory=dict)

    def upsert(self, rule: SemanticRule) -> None:
        self.rules[rule.rule_id] = rule

    def all(self) -> list[SemanticRule]:
        return sorted(self.rules.values(), key=lambda item: item.strength, reverse=True)


@dataclass(slots=True)
class InMemorySemanticGraphStore:
    triples: dict[tuple[str, str, str], KnowledgeTriple] = field(default_factory=dict)

    def upsert(self, triple: KnowledgeTriple) -> None:
        key = (triple.subject, triple.predicate, triple.object)
        current = self.triples.get(key)
        if current is None or current.weight <= triple.weight:
            self.triples[key] = triple

    def all(self) -> list[KnowledgeTriple]:
        return sorted(self.triples.values(), key=lambda item: item.weight, reverse=True)

    def query(self, user_turn: UserTurn, limit: int = 8) -> list[KnowledgeTriple]:
        terms = set(user_turn.user_text.lower().split())
        ranked = sorted(
            self.triples.values(),
            key=lambda triple: (
                len(
                    terms.intersection(
                        f"{triple.subject} {triple.predicate} {triple.object}".lower().split()
                    )
                ),
                triple.weight,
            ),
            reverse=True,
        )
        return ranked[:limit]


from calosum.shared.ports import DatasetExporterPort

@dataclass(slots=True)
class SleepModeConsolidator:
    """
    Converte episodios recentes em memoria semantica mais estavel.

    Em uma implementacao real, este modulo chamaria pipelines analiticos,
    extraindo regras, atualizando um grafo de conhecimento e gerando exemplos
    para adaptacao incremental do SLM via LoRA.
    """

    exporter: DatasetExporterPort | None = None
    minimum_frequency: int = 2

    def consolidate(self, episodes: list[MemoryEpisode]) -> ConsolidationReport:
        started_at = utc_now()
        promoted_rules: list[SemanticRule] = []
        lora_backlog: list[str] = []
        graph_updates: list[KnowledgeTriple] = []

        emotion_counter = Counter[str]()
        support_map: dict[str, list[str]] = {}
        preference_support: dict[str, list[str]] = {}
        
        # Para treinar o LoRA ou DSPy, precisamos gerar um dataset.
        # Vamos acumular os episódios categorizados.
        dspy_dataset: list[dict] = []
        sharegpt_dataset: list[dict] = []

        for episode in episodes:
            for label in episode.right_state.emotional_labels:
                emotion_counter[label] += 1
                support_map.setdefault(label, []).append(episode.episode_id)

            preference = self._extract_preference(episode.user_turn.user_text)
            if preference:
                preference_support.setdefault(preference, []).append(episode.episode_id)
                
            if episode.left_result and episode.left_result.response_text:
                rejected_count = sum(1 for r in episode.execution_results if r.status == "rejected")
                is_corrected = episode.runtime_retry_count > 0 or episode.critique_revision_count > 0
                is_bad = rejected_count > 0
                
                category = "good"
                if is_bad:
                    category = "bad"
                elif is_corrected:
                    category = "corrected"
                
                dspy_dataset.append({
                    "category": category,
                    "input_text": episode.user_turn.user_text,
                    "response_text": episode.left_result.response_text,
                    "runtime_retry_count": episode.runtime_retry_count,
                    "critique_revision_count": episode.critique_revision_count,
                    "actions": [a.action_type for a in episode.left_result.actions]
                })

                # Exporta apenas os sucessos absolutos para o dataset do LoRA no padrão ShareGPT
                if category == "good":
                    sharegpt_dataset.append({
                        "conversations": [
                            {"from": "human", "value": episode.user_turn.user_text},
                            {"from": "gpt", "value": episode.left_result.response_text}
                        ]
                    })

        for label, count in emotion_counter.items():
            if count >= self.minimum_frequency:
                rule = SemanticRule(
                    rule_id=f"emotion::{label}",
                    statement=f"When context shows {label}, bias the response toward empathy and safety.",
                    strength=round(min(1.0, 0.4 + count * 0.15), 2),
                    supporting_episodes=support_map[label],
                    tags=["emotion", "policy"],
                )
                promoted_rules.append(rule)
                lora_backlog.append(f"adapter_rule::{label}")
                graph_updates.append(
                    KnowledgeTriple(
                        subject=f"affect:{label}",
                        predicate="biases_response_toward",
                        object="empathy_and_safety",
                        weight=rule.strength,
                        source_rule_id=rule.rule_id,
                    )
                )

        for preference, episode_ids in preference_support.items():
            if len(episode_ids) >= self.minimum_frequency:
                rule = SemanticRule(
                    rule_id=f"preference::{uuid4()}",
                    statement=preference,
                    strength=0.8,
                    supporting_episodes=episode_ids,
                    tags=["user_preference"],
                )
                promoted_rules.append(rule)
                lora_backlog.append(f"adapter_preference::{preference[:24]}")
                graph_updates.extend(self._preference_to_triples(preference, rule.rule_id))
                
        # Grava os dados limpos no disco para o script noturno do DSPy / PEFT
        if dspy_dataset and self.exporter:
            export_path = self.exporter.export(dspy_dataset, "dspy_dataset.jsonl")
            lora_backlog.append(f"dataset_exported::{export_path}")
            
        if sharegpt_dataset and self.exporter:
            sharegpt_path = self.exporter.export(sharegpt_dataset, "lora_sharegpt.jsonl")
            lora_backlog.append(f"sharegpt_exported::{sharegpt_path}")

        return ConsolidationReport(
            started_at=started_at,
            finished_at=utc_now(),
            episodes_considered=len(episodes),
            promoted_rules=promoted_rules,
            lora_adaptation_backlog=lora_backlog,
            graph_updates=graph_updates,
        )

    def _extract_preference(self, text: str) -> str | None:
        lowered = text.lower()
        markers = ("prefiro", "gosto de", "me chame de")
        for marker in markers:
            if marker in lowered:
                return text.strip()
        return None

    def _preference_to_triples(self, preference: str, rule_id: str) -> list[KnowledgeTriple]:
        lowered = preference.lower()
        triples: list[KnowledgeTriple] = [
            KnowledgeTriple(
                subject="user",
                predicate="expressed_preference",
                object=preference.strip(),
                weight=0.8,
                source_rule_id=rule_id,
            )
        ]
        if "respostas curtas" in lowered or "resposta curta" in lowered:
            triples.append(
                KnowledgeTriple(
                    subject="user",
                    predicate="prefers_response_style",
                    object="short",
                    weight=0.9,
                    source_rule_id=rule_id,
                )
            )
        if "passos claros" in lowered or "plano" in lowered:
            triples.append(
                KnowledgeTriple(
                    subject="user",
                    predicate="prefers_structure",
                    object="stepwise",
                    weight=0.9,
                    source_rule_id=rule_id,
                )
            )
        return triples


@dataclass(slots=True)
class DualMemorySystem:
    episodic_store: EpisodicMemoryStore = field(default_factory=InMemoryEpisodicStore)
    semantic_store: SemanticMemoryStore = field(default_factory=InMemorySemanticStore)
    graph_store: SemanticGraphStore = field(default_factory=InMemorySemanticGraphStore)
    consolidator: SleepModeConsolidator = field(default_factory=SleepModeConsolidator)

    def build_context(self, user_turn: UserTurn, episodic_limit: int = 5) -> MemoryContext:
        return MemoryContext(
            recent_episodes=self.episodic_store.query(user_turn, episodic_limit),
            semantic_rules=self.semantic_store.all(),
            knowledge_triples=self.graph_store.query(user_turn),
        )

    async def abuild_context(self, user_turn: UserTurn, episodic_limit: int = 5) -> MemoryContext:
        return self.build_context(user_turn, episodic_limit)

    def store_episode(self, episode: MemoryEpisode) -> None:
        self.episodic_store.add(episode)

    async def astore_episode(self, episode: MemoryEpisode) -> None:
        self.store_episode(episode)

    def sleep_mode(self) -> ConsolidationReport:
        episodes = self.episodic_store.all()
        report = self.consolidator.consolidate(episodes)
        for rule in report.promoted_rules:
            self.semantic_store.upsert(rule)
        for triple in report.graph_updates:
            self.graph_store.upsert(triple)
        return report

    async def asleep_mode(self) -> ConsolidationReport:
        return self.sleep_mode()
