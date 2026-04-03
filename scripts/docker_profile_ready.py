from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from calosum.bootstrap.entry.api import _build_readiness_payload
from calosum.bootstrap.infrastructure.settings import InfrastructureSettings
from calosum.bootstrap.wiring.factory import CalosumAgentBuilder


def build_docker_readiness() -> dict[str, Any]:
    runtime_root = Path(tempfile.mkdtemp(prefix="calosum-docker-ready-"))
    settings = InfrastructureSettings.from_sources(
        environ={
            "CALOSUM_IGNORE_DOTENV": "true",
            "CALOSUM_MODE": "local",
            "CALOSUM_INFRA_PROFILE": "docker",
            "CALOSUM_VECTOR_QUANTIZATION": "none",
            "CALOSUM_MEMORY_DIR": str(runtime_root / "memory"),
            "CALOSUM_OTLP_JSONL": str(runtime_root / "telemetry" / "events.jsonl"),
            "CALOSUM_DUCKDB_PATH": str(runtime_root / "state" / "semantic.duckdb"),
            "CALOSUM_VECTORDB_URL": "http://127.0.0.1:9",
            "CALOSUM_BRIDGE_STATE_DIR": str(runtime_root / "bridge-state"),
            "CALOSUM_EVOLUTION_ARCHIVE_PATH": str(runtime_root / "evolution" / "archive.jsonl"),
            "CALOSUM_LEFT_ENDPOINT": "http://left.local/v1/chat/completions",
            "CALOSUM_LEFT_BACKEND": "rlm",
            "CALOSUM_RIGHT_BACKEND": "jepars",
            "CALOSUM_RIGHT_JEPARS_BINARY": "jepa-rs",
            "CALOSUM_BRIDGE_BACKEND": "cross_attention",
        }
    )
    builder = CalosumAgentBuilder(settings)
    agent = builder.build()
    return _build_readiness_payload(builder.describe(agent))


def write_outputs(output_json: Path, output_md: Path, payload: dict[str, Any]) -> None:
    stamped = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": "docker",
        "payload": payload,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(stamped, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_md.write_text(
        "\n".join(
            [
                "# Docker Profile Ready Smoke",
                "",
                f"- status: {payload['status']}",
                f"- health: {payload['health']}",
                f"- right_backend: {payload['components']['right_hemisphere']['backend']}",
                f"- left_backend: {payload['components']['left_hemisphere']['backend']}",
                f"- turn_contract: {payload['turn_contract'].get('multi_candidate')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    payload = build_docker_readiness()
    write_outputs(Path(args.output_json), Path(args.output_md), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
