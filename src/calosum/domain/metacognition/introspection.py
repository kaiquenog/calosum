from __future__ import annotations

import uuid
from typing import Any

from calosum.shared.models.types import CognitiveBottleneck, SessionDiagnostic


class IntrospectionEngine:
    """
    Analisa telemetria histórica e produz um diagnóstico grounded no runtime.
    """

    def analyze(
        self,
        session_id: str,
        dashboard_data: dict[str, list[dict[str, Any]]],
        *,
        pending_directive_count: int = 0,
    ) -> SessionDiagnostic:
        decisions = dashboard_data.get("decision", [])
        thoughts = dashboard_data.get("thought", [])
        felt = dashboard_data.get("felt", [])
        execution = dashboard_data.get("execution", [])

        if not decisions:
            return SessionDiagnostic(
                session_id=session_id,
                analyzed_turns=0,
                tool_success_rate=1.0,
                average_retries=0.0,
                average_surprise=0.0,
                bottlenecks=[],
                failure_types={},
                pending_approval_backlog=0,
                pending_directive_count=pending_directive_count,
                surprise_trend=0.0,
            )

        analyzed_turns = len(decisions)
        avg_tool_success = self._average(decisions, "tool_success_rate", default=1.0)
        avg_retries = self._average(decisions, "runtime_retry_count", default=0.0)
        avg_surprise = self._average(felt, "surprise_score", default=0.0)
        surprise_trend = self._surprise_trend(felt)
        dominant_variant, dominant_ratio = self._dominant_variant(thoughts)
        failure_types = self._failure_type_counts(execution)
        approval_backlog = self._approval_backlog(execution)

        bottlenecks = self._identify_bottlenecks(
            avg_tool_success=avg_tool_success,
            avg_retries=avg_retries,
            avg_surprise=avg_surprise,
            surprise_trend=surprise_trend,
            dominant_variant=dominant_variant,
            dominant_ratio=dominant_ratio,
            failure_types=failure_types,
            approval_backlog=approval_backlog,
            pending_directive_count=pending_directive_count,
            decisions=decisions,
        )

        return SessionDiagnostic(
            session_id=session_id,
            analyzed_turns=analyzed_turns,
            tool_success_rate=round(avg_tool_success, 3),
            average_retries=round(avg_retries, 3),
            average_surprise=round(avg_surprise, 3),
            bottlenecks=bottlenecks,
            failure_types=failure_types,
            pending_approval_backlog=approval_backlog,
            pending_directive_count=pending_directive_count,
            surprise_trend=round(surprise_trend, 3),
            dominant_variant=dominant_variant,
            dominant_variant_ratio=round(dominant_ratio, 3),
        )

    def _identify_bottlenecks(
        self,
        *,
        avg_tool_success: float,
        avg_retries: float,
        avg_surprise: float,
        surprise_trend: float,
        dominant_variant: str | None,
        dominant_ratio: float,
        failure_types: dict[str, int],
        approval_backlog: int,
        pending_directive_count: int,
        decisions: list[dict[str, Any]],
    ) -> list[CognitiveBottleneck]:
        bottlenecks: list[CognitiveBottleneck] = []

        if avg_tool_success < 0.7:
            bottlenecks.append(
                self._bottleneck(
                    description="High tool failure rate detected",
                    severity=0.8,
                    evidence=[f"Tool success rate is {avg_tool_success:.1%}, below 70% threshold"],
                    affected_components=["action_runtime", "left_hemisphere"],
                )
            )

        if avg_retries > 1.5:
            bottlenecks.append(
                self._bottleneck(
                    description="Excessive runtime retries",
                    severity=0.6,
                    evidence=[f"Average retries per turn is {avg_retries:.1f}, above 1.5 threshold"],
                    affected_components=["left_hemisphere", "verifier"],
                )
            )

        if approval_backlog > 0:
            bottlenecks.append(
                self._bottleneck(
                    description="Approval backlog accumulating",
                    severity=min(0.9, 0.35 + approval_backlog * 0.1),
                    evidence=[f"Detected {approval_backlog} approval-dependent runtime actions awaiting approval"],
                    affected_components=["action_runtime", "orchestrator"],
                )
            )

        if pending_directive_count > 0:
            bottlenecks.append(
                self._bottleneck(
                    description="Evolution directives awaiting review",
                    severity=min(0.7, 0.2 + pending_directive_count * 0.08),
                    evidence=[f"Detected {pending_directive_count} pending directives requiring operator review"],
                    affected_components=["orchestrator", "reflection_controller"],
                )
            )

        if dominant_variant == "default" and dominant_ratio > 0.9:
            bottlenecks.append(
                self._bottleneck(
                    description="Over-reliance on default reasoning path",
                    severity=0.4,
                    evidence=[f"Variant '{dominant_variant}' dominates {dominant_ratio:.1%} of recent turns"],
                    affected_components=["reflection_controller"],
                )
            )

        if avg_surprise < 0.1 and len(decisions) > 5:
            bottlenecks.append(
                self._bottleneck(
                    description="Agent desensitization",
                    severity=0.5,
                    evidence=[f"Average surprise is very low ({avg_surprise:.2f}) over {len(decisions)} turns"],
                    affected_components=["right_hemisphere"],
                )
            )

        if surprise_trend > 0.12:
            bottlenecks.append(
                self._bottleneck(
                    description="Surprise trend rising across turns",
                    severity=0.55,
                    evidence=[f"Surprise trend is {surprise_trend:+.3f} over the sampled window"],
                    affected_components=["right_hemisphere", "left_hemisphere"],
                )
            )

        structural_failures = {
            name: count
            for name, count in failure_types.items()
            if name in {"runtime_crash", "tool_not_found", "validation_failed"}
        }
        if structural_failures:
            evidence = [
                f"{name}: {count}"
                for name, count in sorted(structural_failures.items())
            ]
            bottlenecks.append(
                self._bottleneck(
                    description="Runtime/tool contract failures recurring",
                    severity=0.65,
                    evidence=evidence,
                    affected_components=["action_runtime", "left_hemisphere", "verifier"],
                )
            )

        return bottlenecks

    def _approval_backlog(self, execution: list[dict[str, Any]]) -> int:
        backlog = 0
        for event in execution:
            for result in event.get("results", []):
                if result.get("status") == "needs_approval":
                    backlog += 1
        return backlog

    def _dominant_variant(self, thoughts: list[dict[str, Any]]) -> tuple[str | None, float]:
        variants = [item.get("active_variant", "default") for item in thoughts if item.get("active_variant")]
        if not variants:
            return None, 0.0

        most_common = max(set(variants), key=variants.count)
        dominance = variants.count(most_common) / len(variants)
        return most_common, dominance

    def _failure_type_counts(self, execution: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in execution:
            for result in event.get("results", []):
                failure_name = None
                if result.get("status") == "needs_approval":
                    failure_name = "approval_required"
                elif result.get("status") == "rejected":
                    output = result.get("output", {})
                    if isinstance(output, dict):
                        failure_name = str(output.get("error_type") or "runtime_rejection")
                    else:
                        failure_name = "runtime_rejection"

                if failure_name is None:
                    continue
                counts[failure_name] = counts.get(failure_name, 0) + 1
        return counts

    def _surprise_trend(self, felt: list[dict[str, Any]]) -> float:
        scores = [float(item.get("surprise_score", 0.0)) for item in felt if "surprise_score" in item]
        if len(scores) < 2:
            return 0.0
        return (scores[-1] - scores[0]) / (len(scores) - 1)

    def _average(
        self,
        rows: list[dict[str, Any]],
        field_name: str,
        *,
        default: float,
    ) -> float:
        if not rows:
            return default
        total = sum(float(row.get(field_name, default)) for row in rows)
        return total / len(rows)

    def _bottleneck(
        self,
        *,
        description: str,
        severity: float,
        evidence: list[str],
        affected_components: list[str],
    ) -> CognitiveBottleneck:
        return CognitiveBottleneck(
            bottleneck_id=str(uuid.uuid4()),
            description=description,
            severity=round(severity, 3),
            evidence=evidence,
            affected_components=affected_components,
        )
