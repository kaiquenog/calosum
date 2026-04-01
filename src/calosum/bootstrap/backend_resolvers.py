from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from calosum.adapters.active_inference import ActiveInferenceRightHemisphereAdapter
from calosum.adapters.bridge_cross_attention import CrossAttentionBridgeAdapter
from calosum.adapters.contract_wrappers import (
    ContractEnforcedLeftHemisphereAdapter,
    ContractEnforcedRightHemisphereAdapter,
)
from calosum.adapters.gea_experience_store import GeaExperienceStoreConfig, SqliteGeaExperienceStore
from calosum.adapters.gea_reflection_experience import ExperienceAwareGEAReflectionController
from calosum.adapters.left_hemisphere_rlm import RlmAdapterConfig, RlmLeftHemisphereAdapter
from calosum.adapters.llm_failover import ResilientLeftHemisphereAdapter
from calosum.adapters.llm_qwen import QwenAdapterConfig, QwenLeftHemisphereAdapter
from calosum.adapters.multimodal_perception import LocalClipVisionAdapter
from calosum.adapters.right_hemisphere_jepars import JepaRsConfig, JepaRsRightHemisphereAdapter
from calosum.adapters.right_hemisphere_vjepa21 import VJepa21Config, VJepa21RightHemisphereAdapter
from calosum.adapters.right_hemisphere_vljepa import VLJepaConfig, VLJepaRightHemisphereAdapter
from calosum.bootstrap.settings import InfrastructureSettings
from calosum.domain.metacognition import GEAReflectionController
from calosum.domain.right_hemisphere import RightHemisphereJEPA

logger = logging.getLogger(__name__)


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
    if not settings.gea_sharing_enabled:
        return GEAReflectionController()

    if settings.gea_experience_store_path is None:
        return GEAReflectionController()

    store = SqliteGeaExperienceStore(
        GeaExperienceStoreConfig(path=settings.gea_experience_store_path)
    )
    return ExperienceAwareGEAReflectionController(experience_store=store)


def resolve_left_hemisphere(
    settings: InfrastructureSettings,
    reason_model_name: str,
) -> tuple[Any, str]:
    backend = (settings.left_hemisphere_backend or "").strip().lower()
    if backend == "rlm":
        adapter = RlmLeftHemisphereAdapter(
            RlmAdapterConfig(
                runtime_command=settings.left_rlm_runtime_command,
                model_path=str(settings.left_rlm_path) if settings.left_rlm_path else None,
                max_depth=settings.left_rlm_max_depth,
            )
        )
        return ContractEnforcedLeftHemisphereAdapter(adapter), "rlm_recursive_adapter"

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
        return (
            ContractEnforcedLeftHemisphereAdapter(
                ResilientLeftHemisphereAdapter([primary, fallback])
            ),
            "resilient_failover_adapter",
        )

    return ContractEnforcedLeftHemisphereAdapter(primary), _legacy_left_backend_name(settings)


def resolve_right_hemisphere(
    settings: InfrastructureSettings,
    vision_adapter: Any | None = None,
    codec: Any | None = None,
) -> tuple[ActiveInferenceRightHemisphereAdapter, str, str]:
    backend = (settings.right_hemisphere_backend or "").strip().lower()
    requested_model = (settings.perception_model or "").strip()

    if backend in {"", "legacy", "auto"}:
        if requested_model.lower() == "jepa":
            base = RightHemisphereJEPA(vision_adapter=vision_adapter)
            setattr(base, "degraded_reason", None)
            return _active_inference_right(base), "active_inference_jepa_policy", "jepa"

        try:
            from calosum.adapters.right_hemisphere_hf import (
                HuggingFaceRightHemisphereAdapter,
                HuggingFaceRightHemisphereConfig,
            )

            cfg = HuggingFaceRightHemisphereConfig(
                embedding_model_name=requested_model or "paraphrase-multilingual-MiniLM-L12-v2"
            )
            base = HuggingFaceRightHemisphereAdapter(cfg, codec=codec)
            return (
                _active_inference_right(base),
                "active_inference_huggingface",
                cfg.embedding_model_name,
            )
        except Exception as exc:
            logger.warning("Falling back to heuristic right hemisphere adapter: %s", exc)
            base = RightHemisphereJEPA(vision_adapter=vision_adapter)
            setattr(base, "degraded_reason", f"hf_stack_unavailable:{exc.__class__.__name__}")
            return (
                _active_inference_right(base),
                "active_inference_heuristic_fallback",
                "jepa",
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
        return _active_inference_right(base), "active_inference_vjepa21", "v-jepa-2.1-local"

    if backend == "vljepa":
        base = VLJepaRightHemisphereAdapter(
            VLJepaConfig(
                model_path=settings.right_model_path,
                action_conditioned=settings.right_action_conditioned,
                horizon=settings.right_horizon,
            ),
            vision_adapter=vision_adapter,
        )
        return _active_inference_right(base), "active_inference_vljepa", "vl-jepa-local"

    if backend == "jepars":
        base = JepaRsRightHemisphereAdapter(
            JepaRsConfig(
                binary_path=settings.right_jepars_binary or "jepa-rs",
                model_path=str(settings.right_model_path) if settings.right_model_path else None,
            )
        )
        return _active_inference_right(base), "active_inference_jepars", "jepa-rs"

    if backend == "huggingface":
        from calosum.adapters.right_hemisphere_hf import (
            HuggingFaceRightHemisphereAdapter,
            HuggingFaceRightHemisphereConfig,
        )

        model_name = requested_model or "paraphrase-multilingual-MiniLM-L12-v2"
        base = HuggingFaceRightHemisphereAdapter(
            HuggingFaceRightHemisphereConfig(embedding_model_name=model_name),
            codec=codec,
        )
        return _active_inference_right(base), "active_inference_huggingface", model_name

    raise ValueError(f"unsupported right hemisphere backend: {backend}")


def _build_qwen(
    *,
    endpoint: str | None,
    api_key: str | None,
    model: str | None,
    provider: str | None,
    reasoning_effort: str | None,
) -> QwenLeftHemisphereAdapter:
    if endpoint:
        return QwenLeftHemisphereAdapter(
            QwenAdapterConfig(
                api_url=endpoint,
                api_key=api_key or "empty",
                model_name=model or "Qwen/Qwen-3.5-9B-Instruct",
                provider=provider or "auto",
                reasoning_effort=reasoning_effort,
            )
        )
    return QwenLeftHemisphereAdapter()


def _active_inference_right(base_adapter: Any) -> ActiveInferenceRightHemisphereAdapter:
    wrapped = ContractEnforcedRightHemisphereAdapter(base_adapter)
    return ActiveInferenceRightHemisphereAdapter(wrapped)


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
