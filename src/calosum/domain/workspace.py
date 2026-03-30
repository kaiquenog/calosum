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
    previous_workspace = agent.last_workspace_by_session.get(user_turn.session_id)
    previous_runtime_feedback = previous_workspace.runtime_feedback[-3:] if previous_workspace else []
    previous_verifier_feedback = previous_workspace.verifier_feedback[-2:] if previous_workspace else []

    return CognitiveWorkspace(
        task_frame={
            "session_id": user_turn.session_id,
            "turn_id": user_turn.turn_id,
            "user_text": user_turn.user_text,
            "previous_runtime_feedback": previous_runtime_feedback,
            "previous_verifier_feedback": previous_verifier_feedback,
        },
        self_model_ref=self_model_ref,
        capability_snapshot=capability_snapshot,
    )
