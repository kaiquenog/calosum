from __future__ import annotations

from calosum.shared.types import UserTurn, utc_now


def build_idle_foraging_turn(memory_system) -> UserTurn:
    session_id = f"idle-foraging-{utc_now().strftime('%Y%m%d%H%M%S')}"
    graph_data = ""
    if hasattr(memory_system, "graph_store"):
        try:
            triples = memory_system.graph_store.all()
            if triples:
                graph_data = " Here are your current top triples: " + "; ".join(
                    f"({t.subject} -> {t.predicate} -> {t.object})" for t in triples[:5]
                )
        except Exception:
            pass

    prompt = (
        "SYSTEM IDLE MODE. You are in an endogenous goal generation state. "
        "Review your current knowledge and project context for gaps, staleness, or ambiguity."
        f"{graph_data} "
        "Actively use your tools (like search_web, execute_bash, read_file) to forage for new, relevant information "
        "and propose plans or semantic rules to reduce future uncertainty."
    )
    return UserTurn(session_id=session_id, user_text=prompt, signals=[])
