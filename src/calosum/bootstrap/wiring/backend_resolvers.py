from __future__ import annotations

import os
from typing import Any

from calosum.adapters.bridge.bridge_cross_attention import CrossAttentionBridgeAdapter
from calosum.adapters.infrastructure.contract_wrappers import (
    ContractEnforcedLeftHemisphereAdapter,
    ContractEnforcedRightHemisphereAdapter,
)
from calosum.domain.metacognition.metacognition import GEAReflectionController
from calosum.adapters.hemisphere.left_hemisphere_rlm_ast import RlmAstAdapterConfig, RlmAstLeftHemisphereAdapter
from calosum.adapters.hemisphere.input_perception_heuristic_jepa import HeuristicJEPAAdapter
from calosum.adapters.hemisphere.input_perception_trained_jepa import TrainedJEPAAdapter
from calosum.adapters.llm.llm_failover import ResilientLeftHemisphereAdapter
from calosum.adapters.llm.llm_fusion import MultiSampleFusionConfig, MultiSampleFusionLeftHemisphereAdapter
from calosum.adapters.llm.llm_qwen import QwenAdapterConfig, QwenLeftHemisphereAdapter
from calosum.adapters.perception.multimodal_perception import LocalClipVisionAdapter
from calosum.adapters.hemisphere.input_perception_jepars import JepaRsConfig, JepaRsRightHemisphereAdapter
from calosum.adapters.hemisphere.input_perception_vjepa21 import VJepa21Config, VJepa21RightHemisphereAdapter
from calosum.adapters.hemisphere.input_perception_vljepa import VLJepaConfig, VLJepaRightHemisphereAdapter
from calosum.bootstrap.infrastructure.settings import CalosumMode, InfrastructureProfile, InfrastructureSettings

def resolve_vision_adapter() -> LocalClipVisionAdapter:
    return LocalClipVisionAdapter()


def resolve_bridge_fusion(settings: InfrastructureSettings):
    backend = (settings.bridge_backend or "").strip().lower()
    if backend in {"", "heuristic", "projection"}:
        return None
    if backend == "cross_attention":
        return CrossAttentionBridgeAdapter()
    raise ValueError(f"unsupported bridge backend: {backend}")


def resolve_reflection_controller(settings: InfrastructureSettings):
    if _env_bool("CALOSUM_GEA_ENABLE_LEARNED_SELECTOR", False):
        from calosum.adapters.experience.gea_reflection_experience import (
            LearnedPreferenceGEAReflectionController,
        )

        return LearnedPreferenceGEAReflectionController()
    return GEAReflectionController()


def resolve_left_hemisphere(
    settings: InfrastructureSettings,
    reason_model_name: str,
) -> tuple[Any, str]:
    backend = (settings.left_hemisphere_backend or "").strip().lower()
    if settings.mode == CalosumMode.API and not settings.left_hemisphere_endpoint:
        raise RuntimeError(
            "CALOSUM_MODE=api requires CALOSUM_LEFT_ENDPOINT; refusing self-referential default."
        )
    if _env_bool("CALOSUM_REQUIRE_LEFT_ENDPOINT", False) and backend != "rlm" and not settings.left_hemisphere_endpoint:
        raise RuntimeError(
            "CALOSUM_REQUIRE_LEFT_ENDPOINT=1 requires CALOSUM_LEFT_ENDPOINT for API-backed left hemisphere backends."
        )

    use_rlm_default = settings.mode == CalosumMode.LOCAL or settings.profile in {
        InfrastructureProfile.PERSISTENT,
        InfrastructureProfile.DOCKER,
    }
    if backend == "rlm" or (backend in {"", "auto"} and use_rlm_default):
        adapter = RlmAstLeftHemisphereAdapter(
            RlmAstAdapterConfig(
                runtime_command=settings.left_rlm_runtime_command,
                model_path=str(settings.left_rlm_path) if settings.left_rlm_path else None,
                max_depth=settings.left_rlm_max_depth,
            )
        )
        backend_name = "rlm_recursive_adapter" if backend == "rlm" else "rlm_recursive_adapter_default"
        return ContractEnforcedLeftHemisphereAdapter(adapter), backend_name

    primary = _build_qwen(
        endpoint=settings.left_hemisphere_endpoint,
        api_key=settings.left_hemisphere_api_key,
        model=reason_model_name,
        provider=settings.left_hemisphere_provider,
        reasoning_effort=settings.left_hemisphere_reasoning_effort,
    )

    if settings.left_hemisphere_fallback_endpoint:
        fallback = _build_qwen(
            endpoint=settings.left_hemisphere_fallback_endpoint,
            api_key=settings.left_hemisphere_fallback_api_key,
            model=settings.left_hemisphere_fallback_model or reason_model_name,
            provider=settings.left_hemisphere_fallback_provider,
            reasoning_effort=(
                settings.left_hemisphere_fallback_reasoning_effort
                or settings.left_hemisphere_reasoning_effort
            ),
        )
        resilient = ResilientLeftHemisphereAdapter([primary, fallback])
        return (
            ContractEnforcedLeftHemisphereAdapter(
                _with_fusion_if_enabled(resilient, settings)
            ),
            "resilient_failover_adapter",
        )

    return ContractEnforcedLeftHemisphereAdapter(
        _with_fusion_if_enabled(primary, settings)
    ), _legacy_left_backend_name(settings)


def resolve_right_hemisphere(
    settings: InfrastructureSettings,
    vision_adapter: Any | None = None,
    codec: Any | None = None,
) -> tuple[Any, str, str]:
    backend = (settings.right_hemisphere_backend or "").strip().lower()
    requested_model = (settings.perception_model or "").strip()

    if backend in {"", "legacy", "auto"}:
        if settings.mode == CalosumMode.LOCAL:
            trained = TrainedJEPAAdapter()
            if trained.is_available:
                return (
                    _active_inference_right(trained),
                    "predictive_checkpoint",
                    trained.config.model_name,
                )
        base = HeuristicJEPAAdapter()
        return (
            _active_inference_right(base),
            "heuristic_literal",
            "heuristic-jepa-phase1",
        )

    if backend == "trained_jepa":
        trained = TrainedJEPAAdapter()
        if trained.is_available:
            return (
                _active_inference_right(trained),
                "predictive_checkpoint",
                trained.config.model_name,
            )
        base = HeuristicJEPAAdapter()
        setattr(base, "degraded_reason", f"trained_jepa_unavailable:{trained.degraded_reason}")
        return (
            _active_inference_right(base),
            "heuristic_literal_fallback",
            "heuristic-jepa-phase1",
        )

    if backend == "heuristic_jepa":
        base = HeuristicJEPAAdapter()
        return (
            _active_inference_right(base),
            "heuristic_literal",
            "heuristic-jepa-phase1",
        )

    if backend == "vjepa21":
        base = VJepa21RightHemisphereAdapter(
            VJepa21Config(
                model_path=settings.right_model_path,
                action_conditioned=settings.right_action_conditioned,
                horizon=settings.right_horizon,
            ),
            vision_adapter=vision_adapter,
            codec=codec,
        )
        return _active_inference_right(base), "vjepa21_local", "v-jepa-2.1-local"

    if backend == "vljepa":
        base = VLJepaRightHemisphereAdapter(
            VLJepaConfig(
                model_path=settings.right_model_path,
                action_conditioned=settings.right_action_conditioned,
                horizon=settings.right_horizon,
            ),
            vision_adapter=vision_adapter,
        )
        return _active_inference_right(base), "vljepa_local", "vl-jepa-local"

    if backend == "jepars":
        base = JepaRsRightHemisphereAdapter(
            JepaRsConfig(
                binary_path=settings.right_jepars_binary or "jepa-rs",
                model_path=str(settings.right_model_path) if settings.right_model_path else None,
            )
        )
        return _active_inference_right(base), "jepars_local", "jepa-rs"

    if backend == "huggingface":
        from calosum.adapters.hemisphere.input_perception_hf import (
            HuggingFaceRightHemisphereAdapter,
            HuggingFaceRightHemisphereConfig,
        )

        model_name = requested_model or "paraphrase-multilingual-MiniLM-L12-v2"
        base = HuggingFaceRightHemisphereAdapter(
            HuggingFaceRightHemisphereConfig(embedding_model_name=model_name),
            codec=codec,
        )
        return _active_inference_right(base), "distance_huggingface", model_name

    raise ValueError(f"unsupported right hemisphere backend: {backend}")


def _build_qwen(
    *,
    endpoint: str | None,
    api_key: str | None,
    model: str | None,
    provider: str | None,
    reasoning_effort: str | None,
) -> QwenLeftHemisphereAdapter:
    return QwenLeftHemisphereAdapter(
        QwenAdapterConfig(
            api_url=endpoint,
            api_key=api_key or "empty",
            model_name=model or "Qwen/Qwen-3.5-9B-Instruct",
            provider=provider or "auto",
            reasoning_effort=reasoning_effort,
        )
    )


def _active_inference_right(base_adapter: Any) -> Any:
    return ContractEnforcedRightHemisphereAdapter(base_adapter)


def _with_fusion_if_enabled(provider: Any, settings: InfrastructureSettings) -> Any:
    enabled_default = settings.profile != InfrastructureProfile.EPHEMERAL
    enabled = _env_bool("CALOSUM_FUSION_ENABLED", enabled_default)
    if not enabled:
        return provider
    candidates = max(1, _env_int("CALOSUM_FUSION_CANDIDATES", 3))
    threshold = _env_float("CALOSUM_FUSION_UNCERTAINTY_THRESHOLD", 0.5)
    mode_raw = os.getenv("CALOSUM_FUSION_SELECTION_MODE", "guided").strip().lower()
    mode = "random" if mode_raw in {"random", "control_b", "treatment_b"} else "guided"
    config = MultiSampleFusionConfig(
        enabled=enabled,
        n_candidates=candidates,
        uncertainty_threshold=max(0.0, min(1.0, threshold)),
        selection_mode=mode,
    )
    return MultiSampleFusionLeftHemisphereAdapter(provider=provider, config=config)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _legacy_left_backend_name(settings: InfrastructureSettings) -> str:
    if settings.left_hemisphere_fallback_endpoint:
        return "resilient_failover_adapter"
    endpoint = (settings.left_hemisphere_endpoint or "").lower()
    provider = (settings.left_hemisphere_provider or "auto").lower()

    if provider in {"openai_responses", "openai", "responses"}:
        return "openai_responses_adapter"
    if provider in {"openai_chat", "chat"}:
        return "openai_chat_adapter"
    if provider == "openrouter":
        return "openrouter_adapter"
    if "api.openai.com" in endpoint:
        if endpoint.rstrip("/").endswith("/chat/completions"):
            return "openai_chat_adapter"
        return "openai_responses_adapter"
    if endpoint:
        return "openai_compatible_chat_adapter"
    return "openai_compatible_chat_adapter_default"
