from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VARIANT_LABEL_MAP = {
    "analitico": 0,
    "empatico": 1,
    "pragmatico": 2,
}
LABEL_VARIANT_MAP = {value: key for key, value in VARIANT_LABEL_MAP.items()}
INTENT_LABEL_MAP = {
    "factual": 0,
    "technical": 1,
    "emotional": 2,
    "creative": 3,
}


@dataclass(slots=True)
class PreferenceFeatures:
    surprise_score: float
    ambiguity_score: float
    intent_type: str
    session_length: int
    avg_tool_success_rate: float
    jepa_uncertainty: float


@dataclass(slots=True)
class VariantTrainingExample:
    session_id: str
    turn_id: str
    recorded_at: str
    variant_scores: dict[str, float]
    selected_variant: str
    response_rating: float
    context: dict[str, Any]


@dataclass(slots=True)
class PreferenceTrainingReport:
    trained: bool
    sample_count: int
    holdout_accuracy: float
    reason: str | None = None


class VariantPreferenceDatasetStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, example: VariantTrainingExample) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")

    def read_all(self) -> list[VariantTrainingExample]:
        if not self.path.exists():
            return []
        rows: list[VariantTrainingExample] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                rows.append(
                    VariantTrainingExample(
                        session_id=str(payload.get("session_id", "")),
                        turn_id=str(payload.get("turn_id", "")),
                        recorded_at=str(payload.get("recorded_at", _utc_now_iso())),
                        variant_scores={
                            str(k): float(v)
                            for k, v in (payload.get("variant_scores", {}) or {}).items()
                        },
                        selected_variant=str(payload.get("selected_variant", "")),
                        response_rating=float(payload.get("response_rating", 0.0)),
                        context=dict(payload.get("context", {}) or {}),
                    )
                )
        return rows

    def count(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())


class VariantPreferenceModel:
    def __init__(
        self,
        *,
        artifact_path: Path,
        min_samples: int = 200,
        holdout_ratio: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.artifact_path = Path(artifact_path)
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self.min_samples = max(1, int(min_samples))
        self.holdout_ratio = min(0.4, max(0.1, float(holdout_ratio)))
        self.random_state = int(random_state)

    def train(self, rows: list[VariantTrainingExample]) -> PreferenceTrainingReport:
        examples = self._prepare_examples(rows)
        if len(examples) < self.min_samples:
            return PreferenceTrainingReport(
                trained=False,
                sample_count=len(examples),
                holdout_accuracy=0.0,
                reason=f"insufficient_samples:{len(examples)}<{self.min_samples}",
            )

        estimator = self._build_lightgbm_classifier()
        if estimator is None:
            return PreferenceTrainingReport(
                trained=False,
                sample_count=len(examples),
                holdout_accuracy=0.0,
                reason="lightgbm_unavailable",
            )
        try:
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import train_test_split
        except Exception:
            return PreferenceTrainingReport(
                trained=False,
                sample_count=len(examples),
                holdout_accuracy=0.0,
                reason="sklearn_unavailable",
            )

        x_data = [item[0] for item in examples]
        y_data = [item[1] for item in examples]
        sample_weights = [item[2] for item in examples]

        x_train, x_test, y_train, y_test, w_train, _w_test = train_test_split(
            x_data,
            y_data,
            sample_weights,
            test_size=self.holdout_ratio,
            random_state=self.random_state,
            stratify=y_data if len(set(y_data)) > 1 else None,
        )

        estimator.fit(x_train, y_train, sample_weight=w_train)
        predictions = estimator.predict(x_test)
        accuracy = float(accuracy_score(y_test, predictions))

        metadata = {
            "trained_at": _utc_now_iso(),
            "sample_count": len(examples),
            "min_samples": self.min_samples,
            "holdout_accuracy": round(accuracy, 4),
            "holdout_ratio": self.holdout_ratio,
            "model_type": "LightGBM",
            "feature_order": [
                "surprise_score",
                "ambiguity_score",
                "intent_type",
                "session_length",
                "avg_tool_success_rate",
                "jepa_uncertainty",
            ],
        }
        if not _joblib_dump({"model": estimator, "metadata": metadata}, self.artifact_path):
            return PreferenceTrainingReport(
                trained=False,
                sample_count=len(examples),
                holdout_accuracy=accuracy,
                reason="joblib_unavailable",
            )

        return PreferenceTrainingReport(
            trained=True,
            sample_count=len(examples),
            holdout_accuracy=accuracy,
        )

    def predict(self, features: PreferenceFeatures) -> tuple[str | None, dict[str, Any]]:
        bundle = self._load_bundle()
        if bundle is None:
            return None, {"reason": "model_not_available"}
        model = bundle.get("model")
        metadata = dict(bundle.get("metadata", {}) or {})
        if model is None:
            return None, {"reason": "invalid_artifact"}
        row = [self._feature_row(features)]
        try:
            label = int(model.predict(row)[0])
        except Exception:
            return None, {"reason": "predict_failed", "metadata": metadata}
        variant = LABEL_VARIANT_MAP.get(label)
        return variant, {"reason": "ok", "metadata": metadata}

    def metadata(self) -> dict[str, Any]:
        bundle = self._load_bundle()
        if bundle is None:
            return {}
        return dict(bundle.get("metadata", {}) or {})

    def _prepare_examples(self, rows: list[VariantTrainingExample]) -> list[tuple[list[float], int, float]]:
        prepared: list[tuple[list[float], int, float]] = []
        for row in rows:
            variant = canonical_variant_id(row.selected_variant)
            label = VARIANT_LABEL_MAP.get(variant)
            if label is None:
                continue
            context = row.context or {}
            features = PreferenceFeatures(
                surprise_score=float(context.get("surprise_score", 0.0)),
                ambiguity_score=float(context.get("ambiguity_score", 0.0)),
                intent_type=str(context.get("intent_type", "factual")),
                session_length=max(1, int(context.get("session_length", 1))),
                avg_tool_success_rate=float(context.get("avg_tool_success_rate", 1.0)),
                jepa_uncertainty=float(context.get("jepa_uncertainty", 0.0)),
            )
            weight = max(0.1, min(1.0, float(row.response_rating)))
            prepared.append((self._feature_row(features), label, weight))
        return prepared

    def _feature_row(self, features: PreferenceFeatures) -> list[float]:
        intent_code = INTENT_LABEL_MAP.get(features.intent_type, 0)
        return [
            _clamp(features.surprise_score),
            _clamp(features.ambiguity_score),
            float(intent_code),
            float(max(1, features.session_length)),
            _clamp(features.avg_tool_success_rate),
            _clamp(features.jepa_uncertainty),
        ]

    def _build_lightgbm_classifier(self) -> Any | None:
        try:
            from lightgbm import LGBMClassifier
        except Exception:
            return None
        return LGBMClassifier(
            n_estimators=50,
            max_depth=4,
            random_state=self.random_state,
        )

    def _load_bundle(self) -> dict[str, Any] | None:
        if not self.artifact_path.exists():
            return None
        loaded = _joblib_load(self.artifact_path)
        if loaded is None:
            return None
        return loaded if isinstance(loaded, dict) else None


def canonical_variant_id(raw_variant: str) -> str:
    value = (raw_variant or "").strip().lower()
    if value.startswith("analitico"):
        return "analitico"
    if value.startswith("empatico"):
        return "empatico"
    if value.startswith("pragmatico"):
        return "pragmatico"
    return value


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _joblib_dump(payload: dict[str, Any], path: Path) -> bool:
    try:
        import joblib
    except Exception:
        return False
    joblib.dump(payload, path)
    return True


def _joblib_load(path: Path) -> dict[str, Any] | None:
    try:
        import joblib
    except Exception:
        return None
    try:
        loaded = joblib.load(path)
    except Exception:
        return None
    return loaded if isinstance(loaded, dict) else None
