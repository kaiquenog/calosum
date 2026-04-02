"""Helpers de serialização/desserialização para o QdrantDualMemoryAdapter.

Extraídos de memory_qdrant.py (Sprint 1.5) para manter esse módulo abaixo
de 400 linhas e facilitar testes unitários isolados.
"""
from __future__ import annotations

import base64
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from calosum.shared.ports import VectorCodecPort

from calosum.shared.types import (
    BridgeControlSignal,
    CognitiveBridgePacket,
    LeftHemisphereResult,
    MemoryEpisode,
    PrimitiveAction,
    RightHemisphereState,
    SemanticRule,
    TypedLambdaProgram,
    UserTurn,
    utc_now,
)


# ---------------------------------------------------------------------------
# Public serializers
# ---------------------------------------------------------------------------

def episode_payload(episode: MemoryEpisode) -> dict:
    """Serialize a MemoryEpisode to a Qdrant point payload dict."""
    return {
        "text": episode.user_turn.user_text,
        "session": episode.user_turn.session_id,
        "observed_at": episode.user_turn.observed_at.isoformat(),
        "recorded_at": episode.recorded_at.isoformat(),
        "emotional_labels": episode.right_state.emotional_labels if episode.right_state else [],
        "latent_vector": episode.right_state.latent_vector if episode.right_state else [],
        "salience": episode.right_state.salience if episode.right_state else 0.15,
        "confidence": episode.right_state.confidence if episode.right_state else 0.0,
        "surprise_score": episode.right_state.surprise_score if episode.right_state else 0.0,
        "world_hypotheses": episode.right_state.world_hypotheses if episode.right_state else {},
        "response_text": episode.left_result.response_text if episode.left_result else "",
        "action_types": (
            [action.action_type for action in episode.left_result.actions]
            if episode.left_result
            else []
        ),
        "reasoning_summary": episode.left_result.reasoning_summary if episode.left_result else [],
    }


def episode_from_point(point, codec: VectorCodecPort | None = None) -> MemoryEpisode:
    """Deserialize a Qdrant ScoredPoint / Record into a MemoryEpisode."""
    payload = point.payload or {}
    
    latent_vector = list(payload.get("latent_vector", []))
    compressed = payload.get("latent_vector_compressed")
    
    if codec is not None and compressed:
        try:
            decoded = codec.decode(base64.b64decode(compressed))
            # Se decodificou com sucesso, priorizamos o vetor decodificado (mais preciso/fiel ao original)
            latent_vector = list(decoded)
        except Exception:
            pass

    user_turn = UserTurn(
        session_id=payload.get("session", "*"),
        user_text=payload.get("text", ""),
        signals=[],
        observed_at=_parse_datetime(payload.get("observed_at")),
    )
    right_state = _placeholder_right_state(
        user_turn,
        latent_vector=latent_vector,
        emotional_labels=list(payload.get("emotional_labels", [])),
        salience=float(payload.get("salience", 0.5 if payload.get("emotional_labels") else 0.15)),
        confidence=float(payload.get("confidence", 0.0)),
        surprise_score=float(payload.get("surprise_score", 0.0)),
        world_hypotheses=dict(payload.get("world_hypotheses", {})),
    )
    return MemoryEpisode(
        episode_id=str(point.id),
        recorded_at=_parse_datetime(payload.get("recorded_at")),
        user_turn=user_turn,
        right_state=right_state,
        bridge_packet=_placeholder_bridge_packet(right_state),
        left_result=_placeholder_left_result(
            response_text=payload.get("response_text", ""),
            action_types=list(payload.get("action_types", [])),
            reasoning_summary=list(payload.get("reasoning_summary", [])),
        ),
    )


def rule_from_point(point) -> SemanticRule:
    """Deserialize a Qdrant point into a SemanticRule."""
    payload = point.payload or {}
    return SemanticRule(
        rule_id=payload.get("rule_id", str(point.id)),
        statement=payload.get("statement", ""),
        strength=float(payload.get("strength", 1.0)),
        supporting_episodes=list(payload.get("supporting_episodes", [])),
        tags=list(payload.get("tags", [])),
    )


def rule_document(rule: SemanticRule) -> str:
    """Produce the text document used as embedding input for a SemanticRule."""
    return " ".join(
        part
        for part in [
            rule.statement,
            " ".join(rule.tags),
            " ".join(rule.supporting_episodes),
        ]
        if part
    ).strip()


def episode_document(episode: MemoryEpisode) -> str:
    """Produce the text document used as embedding input for a MemoryEpisode."""
    parts = [
        episode.user_turn.user_text,
        " ".join(episode.right_state.emotional_labels),
        episode.left_result.response_text,
        " ".join(action.action_type for action in episode.left_result.actions),
        " ".join(episode.left_result.reasoning_summary),
    ]
    return " ".join(part for part in parts if part).strip()


# ---------------------------------------------------------------------------
# Internal helpers (reusable by callers)
# ---------------------------------------------------------------------------

def _parse_datetime(value: str | None) -> datetime:
    if not value:
        return utc_now()
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return utc_now()


def _placeholder_right_state(
    user_turn: UserTurn,
    *,
    latent_vector: list[float] | None = None,
    emotional_labels: list[str] | None = None,
    salience: float = 0.15,
    confidence: float = 0.0,
    surprise_score: float = 0.0,
    world_hypotheses: dict | None = None,
) -> RightHemisphereState:
    return RightHemisphereState(
        context_id=user_turn.turn_id,
        latent_vector=latent_vector or [],
        salience=salience,
        emotional_labels=emotional_labels or ["neutral"],
        world_hypotheses=world_hypotheses or {},
        confidence=confidence,
        surprise_score=surprise_score,
        telemetry={"source": "qdrant_placeholder"},
    )


def _placeholder_bridge_packet(right_state: RightHemisphereState) -> CognitiveBridgePacket:
    return CognitiveBridgePacket(
        context_id=right_state.context_id,
        soft_prompts=[],
        control=BridgeControlSignal(
            target_temperature=0.25,
            empathy_priority=right_state.salience >= 0.7,
            system_directives=[],
            annotations={"source": "qdrant_placeholder"},
        ),
        salience=right_state.salience,
        bridge_metadata={"source": "qdrant_placeholder"},
    )


def _placeholder_left_result(
    *,
    response_text: str = "",
    action_types: list[str] | None = None,
    reasoning_summary: list[str] | None = None,
) -> LeftHemisphereResult:
    actions = []
    for action_type in action_types or []:
        actions.append(
            PrimitiveAction(
                action_type=action_type,
                typed_signature="Placeholder -> Placeholder",
                payload={},
                safety_invariants=["placeholder reconstructed from qdrant payload"],
            )
        )
    return LeftHemisphereResult(
        response_text=response_text,
        lambda_program=TypedLambdaProgram(
            signature="Placeholder",
            expression="",
            expected_effect="hydrate memory episode without executable actions",
        ),
        actions=actions,
        reasoning_summary=reasoning_summary or ["placeholder_memory_episode"],
        telemetry={"source": "qdrant_placeholder"},
    )


__all__ = [
    "episode_payload",
    "episode_from_point",
    "rule_from_point",
    "rule_document",
    "episode_document",
    "_parse_datetime",
    "_placeholder_right_state",
    "_placeholder_bridge_packet",
    "_placeholder_left_result",
]
