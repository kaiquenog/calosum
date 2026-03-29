from __future__ import annotations

from typing import TYPE_CHECKING
from calosum.shared.types import CognitiveWorkspace

if TYPE_CHECKING:
    from calosum.domain.orchestrator import CalosumAgent
    from calosum.shared.types import UserTurn


def init_turn_workspace(agent: "CalosumAgent", user_turn: "UserTurn") -> CognitiveWorkspace:
    """
    Inicializa o workspace cognitivo compartilhado para o turno atual.
    """
    from dataclasses import asdict

    self_model_ref = asdict(agent.self_model) if hasattr(agent, "self_model") and agent.self_model else None
    capability_snapshot = agent.capability_snapshot

    return CognitiveWorkspace(
        task_frame={
            "session_id": user_turn.session_id,
            "turn_id": user_turn.turn_id,
            "user_text": user_turn.user_text,
        },
        self_model_ref=self_model_ref,
        capability_snapshot=capability_snapshot,
    )
