from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from calosum.domain.memory.memory import DualMemorySystem
from calosum.shared.utils.serialization import to_primitive
from calosum.shared.models.types import (
    BridgeControlSignal,
    CognitiveBridgePacket,
    KnowledgeTriple,
    LeftHemisphereResult,
    MemoryEpisode,
    Modality,
    MultimodalSignal,
    PrimitiveAction,
    RightHemisphereState,
    SemanticRule,
    SoftPromptToken,
    TypedLambdaProgram,
    UserTurn,
)


@dataclass(slots=True)
class JsonlEpisodicStore:
    path: Path
    episodes: list[MemoryEpisode] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.episodes = [
                _memory_episode_from_dict(json.loads(line))
                for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    def add(self, episode: MemoryEpisode) -> None:
        self.episodes.append(episode)
        self._append_record(episode)

    def query(self, user_turn: UserTurn, limit: int = 5) -> list[MemoryEpisode]:
        query_terms = set(user_turn.user_text.lower().split())
        session_episodes = [
            episode
            for episode in self.episodes
            if episode.user_turn.session_id == user_turn.session_id
        ]
        pool = session_episodes or self.episodes
        ranked = sorted(
            pool,
            key=lambda episode: (
                len(query_terms.intersection(episode.user_turn.user_text.lower().split())),
                episode.recorded_at,
            ),
            reverse=True,
        )
        return ranked[:limit]

    def all(self) -> list[MemoryEpisode]:
        return list(self.episodes)

    def _append_record(self, value: MemoryEpisode) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_primitive(value), ensure_ascii=False, sort_keys=True) + "\n")


@dataclass(slots=True)
class JsonlSemanticStore:
    path: Path
    rules: dict[str, SemanticRule] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rule = _semantic_rule_from_dict(json.loads(line))
                    self.rules[rule.rule_id] = rule

    def upsert(self, rule: SemanticRule) -> None:
        self.rules[rule.rule_id] = rule
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_primitive(rule), ensure_ascii=False, sort_keys=True) + "\n")

    def all(self) -> list[SemanticRule]:
        return sorted(self.rules.values(), key=lambda item: item.strength, reverse=True)


@dataclass(slots=True)
class JsonlSemanticGraphStore:
    path: Path
    triples: dict[tuple[str, str, str], KnowledgeTriple] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    triple = _knowledge_triple_from_dict(json.loads(line))
                    self.triples[(triple.subject, triple.predicate, triple.object)] = triple

    def upsert(self, triple: KnowledgeTriple) -> None:
        key = (triple.subject, triple.predicate, triple.object)
        current = self.triples.get(key)
        if current is None or current.weight <= triple.weight:
            self.triples[key] = triple
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(to_primitive(triple), ensure_ascii=False, sort_keys=True) + "\n"
                )

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


@dataclass(slots=True)
class PersistentDualMemorySystem(DualMemorySystem):
    @classmethod
    def from_directory(cls, base_path: str | Path, consolidator=None) -> "PersistentDualMemorySystem":
        base_dir = Path(base_path)
        base_dir.mkdir(parents=True, exist_ok=True)
        kwargs = {
            "episodic_store": JsonlEpisodicStore(base_dir / "episodic.jsonl"),
            "semantic_store": JsonlSemanticStore(base_dir / "semantic_rules.jsonl"),
            "graph_store": JsonlSemanticGraphStore(base_dir / "knowledge_graph.jsonl"),
        }
        if consolidator is not None:
            kwargs["consolidator"] = consolidator
        return cls(**kwargs)


def _datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _signal_from_dict(data: dict) -> MultimodalSignal:
    return MultimodalSignal(
        modality=Modality(data["modality"]),
        source=data["source"],
        payload=data.get("payload"),
        quality=data.get("quality", 1.0),
        metadata=data.get("metadata", {}),
    )


def _user_turn_from_dict(data: dict) -> UserTurn:
    return UserTurn(
        session_id=data["session_id"],
        user_text=data["user_text"],
        signals=[_signal_from_dict(item) for item in data.get("signals", [])],
        observed_at=_datetime(data["observed_at"]),
        turn_id=data["turn_id"],
    )


def _right_state_from_dict(data: dict) -> RightHemisphereState:
    return RightHemisphereState(
        context_id=data["context_id"],
        latent_vector=data.get("latent_vector", []),
        salience=data["salience"],
        emotional_labels=data.get("emotional_labels", []),
        world_hypotheses=data.get("world_hypotheses", {}),
        confidence=data["confidence"],
        telemetry=data.get("telemetry", {}),
    )


def _soft_prompt_from_dict(data: dict) -> SoftPromptToken:
    return SoftPromptToken(
        token=data["token"],
        weight=data["weight"],
        provenance=data["provenance"],
    )


def _bridge_packet_from_dict(data: dict) -> CognitiveBridgePacket:
    return CognitiveBridgePacket(
        context_id=data["context_id"],
        soft_prompts=[_soft_prompt_from_dict(item) for item in data.get("soft_prompts", [])],
        control=BridgeControlSignal(
            target_temperature=data["control"]["target_temperature"],
            empathy_priority=data["control"]["empathy_priority"],
            system_directives=data["control"].get("system_directives", []),
            annotations=data["control"].get("annotations", {}),
        ),
        salience=data["salience"],
        bridge_metadata=data.get("bridge_metadata", {}),
    )


def _primitive_action_from_dict(data: dict) -> PrimitiveAction:
    return PrimitiveAction(
        action_type=data["action_type"],
        typed_signature=data["typed_signature"],
        payload=data.get("payload", {}),
        safety_invariants=data.get("safety_invariants", []),
    )


def _left_result_from_dict(data: dict) -> LeftHemisphereResult:
    return LeftHemisphereResult(
        response_text=data["response_text"],
        lambda_program=TypedLambdaProgram(
            signature=data["lambda_program"]["signature"],
            expression=data["lambda_program"]["expression"],
            expected_effect=data["lambda_program"]["expected_effect"],
        ),
        actions=[_primitive_action_from_dict(item) for item in data.get("actions", [])],
        reasoning_summary=data.get("reasoning_summary", []),
        telemetry=data.get("telemetry", {}),
    )


def _memory_episode_from_dict(data: dict) -> MemoryEpisode:
    user_turn = _user_turn_from_dict(data["user_turn"])
    right_state_data = data.get("right_state")
    right_state = (
        _right_state_from_dict(right_state_data)
        if right_state_data
        else _placeholder_right_state(user_turn)
    )
    bridge_packet_data = data.get("bridge_packet")
    left_result_data = data.get("left_result")
    return MemoryEpisode(
        episode_id=data["episode_id"],
        recorded_at=_datetime(data["recorded_at"]),
        user_turn=user_turn,
        right_state=right_state,
        bridge_packet=(
            _bridge_packet_from_dict(bridge_packet_data)
            if bridge_packet_data
            else _placeholder_bridge_packet(right_state)
        ),
        left_result=(
            _left_result_from_dict(left_result_data)
            if left_result_data
            else _placeholder_left_result()
        ),
    )


def _semantic_rule_from_dict(data: dict) -> SemanticRule:
    return SemanticRule(
        rule_id=data["rule_id"],
        statement=data["statement"],
        strength=data["strength"],
        supporting_episodes=data.get("supporting_episodes", []),
        tags=data.get("tags", []),
    )


def _knowledge_triple_from_dict(data: dict) -> KnowledgeTriple:
    return KnowledgeTriple(
        subject=data["subject"],
        predicate=data["predicate"],
        object=data["object"],
        weight=data.get("weight", 1.0),
        source_rule_id=data.get("source_rule_id"),
    )


def _placeholder_right_state(user_turn: UserTurn) -> RightHemisphereState:
    return RightHemisphereState(
        context_id=user_turn.turn_id,
        latent_vector=[],
        salience=0.15,
        emotional_labels=["neutral"],
        world_hypotheses={},
        confidence=0.0,
        telemetry={"source": "persistent_memory_placeholder"},
    )


def _placeholder_bridge_packet(right_state: RightHemisphereState) -> CognitiveBridgePacket:
    return CognitiveBridgePacket(
        context_id=right_state.context_id,
        soft_prompts=[],
        control=BridgeControlSignal(
            target_temperature=0.25,
            empathy_priority=False,
            system_directives=[],
            annotations={"source": "persistent_memory_placeholder"},
        ),
        salience=right_state.salience,
        bridge_metadata={"source": "persistent_memory_placeholder"},
    )


def _placeholder_left_result() -> LeftHemisphereResult:
    return LeftHemisphereResult(
        response_text="",
        lambda_program=TypedLambdaProgram(
            signature="Placeholder",
            expression="",
            expected_effect="hydrate persisted episode without executable actions",
        ),
        actions=[],
        reasoning_summary=["placeholder_memory_episode"],
        telemetry={"source": "persistent_memory_placeholder"},
    )
