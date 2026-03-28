from __future__ import annotations

from dataclasses import asdict
from pprint import pprint

from calosum import CalosumAgent, CognitiveVariantSpec, Modality, MultimodalSignal, UserTurn


def main() -> None:
    agent = CalosumAgent()
    turn = UserTurn(
        session_id="group-session",
        user_text="Estou ansioso e preciso de ajuda urgente, mas prefiro respostas curtas.",
        signals=[
            MultimodalSignal(
                modality=Modality.AUDIO,
                source="microphone",
                payload={"transcript": "fala acelerada"},
                metadata={"emotion": "ansioso"},
            ),
            MultimodalSignal(
                modality=Modality.TYPING,
                source="keyboard",
                payload={"cadence": "burst"},
                metadata={"emotion": "frustrado"},
            ),
        ],
    )

    variants = [
        CognitiveVariantSpec(
            variant_id="empathetic_low_threshold",
            tokenizer_overrides={"salience_threshold": 0.45},
            notes=["favor empathy under moderate salience"],
        ),
        CognitiveVariantSpec(
            variant_id="strict_high_threshold",
            tokenizer_overrides={"salience_threshold": 0.9},
            notes=["activate empathy only under stronger affective evidence"],
        ),
    ]

    group_result = agent.process_group_turn(turn, variants)

    print("Vencedor:")
    print(group_result.reflection.selected_variant_id)
    print("\nScoreboard:")
    pprint([asdict(item) for item in group_result.reflection.scoreboard])
    print("\nAjustes no corpo caloso:")
    pprint(group_result.reflection.bridge_adjustments)
    print("\nDashboard:")
    pprint(agent.cognitive_dashboard(turn.session_id))


if __name__ == "__main__":
    main()
