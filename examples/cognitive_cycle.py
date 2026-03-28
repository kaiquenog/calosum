from __future__ import annotations

from dataclasses import asdict
from pprint import pprint

from calosum import CalosumAgent, Modality, MultimodalSignal, UserTurn


def main() -> None:
    agent = CalosumAgent()

    first_turn = UserTurn(
        session_id="demo-session",
        user_text="Estou frustrado e preciso de um plano urgente para reorganizar este projeto.",
        signals=[
            MultimodalSignal(
                modality=Modality.AUDIO,
                source="microphone",
                payload={"transcript": "voz trêmula"},
                metadata={"emotion": "frustrado", "prosody": "tense"},
            ),
            MultimodalSignal(
                modality=Modality.TYPING,
                source="keyboard",
                payload={"cadence": "fast_bursty"},
                metadata={"emotion": "ansioso"},
            ),
        ],
    )

    second_turn = UserTurn(
        session_id="demo-session",
        user_text="Prefiro respostas curtas com passos claros quando a situacao estiver urgente.",
        signals=[],
    )

    for turn in (first_turn, second_turn):
        result = agent.process_turn(turn)
        print(f"\nTurno: {turn.turn_id}")
        print("Resposta:")
        print(result.left_result.response_text)
        print("Soft prompts:")
        pprint([asdict(token) for token in result.bridge_packet.soft_prompts])
        print("Telemetria:")
        pprint(asdict(result.telemetry))

    report = agent.sleep_mode()
    print("\nSleep mode:")
    pprint(asdict(report))


if __name__ == "__main__":
    main()
