import json
from pathlib import Path
from typing import Any

from calosum.shared.models.ports import BridgeStateStorePort


class LocalBridgeStateStore(BridgeStateStorePort):
    def __init__(
        self,
        *,
        base_dir: Path | None = None,
        weights_path: Path | None = None,
        adaptation_path: Path | None = None,
        reflection_history_path: Path | None = None,
    ) -> None:
        root = Path(base_dir) if base_dir is not None else Path(".calosum-runtime/state")
        self.weights_path = Path(weights_path) if weights_path is not None else root / "bridge_weights.pt"
        self.adaptation_path = Path(adaptation_path) if adaptation_path is not None else root / "bridge_config.json"
        self.reflection_history_path = (
            Path(reflection_history_path)
            if reflection_history_path is not None
            else root / "bridge_reflections.jsonl"
        )

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
