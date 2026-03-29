import json
from pathlib import Path
from typing import Any

from calosum.shared.ports import BridgeStateStorePort


class LocalBridgeStateStore(BridgeStateStorePort):
    def __init__(
        self,
        weights_path: Path = Path(".calosum-runtime/state/bridge_weights.pt"),
        adaptation_path: Path = Path(".calosum-runtime/state/bridge_config.json"),
        reflection_history_path: Path = Path(".calosum-runtime/state/bridge_reflections.jsonl"),
    ) -> None:
        self.weights_path = weights_path
        self.adaptation_path = adaptation_path
        self.reflection_history_path = reflection_history_path

    def load_weights(self, projection_layer: Any) -> bool:
        if not self.weights_path.exists():
            return False
        import torch
        projection_layer.load_state_dict(torch.load(self.weights_path, weights_only=True))
        return True

    def load_adaptation_state(self) -> dict[str, Any]:
        if not self.adaptation_path.exists():
            return {}
        try:
            return json.loads(self.adaptation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def persist_adaptation_state(self, state: dict[str, Any]) -> None:
        self.adaptation_path.parent.mkdir(parents=True, exist_ok=True)
        self.adaptation_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def record_reflection_event(self, payload: dict[str, Any]) -> None:
        self.reflection_history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.reflection_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
