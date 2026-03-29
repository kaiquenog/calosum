# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
PYTHONPATH=src python3 -m unittest discover -s tests -t .

# Run a single test file
PYTHONPATH=src python3 -m unittest tests.test_runtime

# Run a single test method
PYTHONPATH=src python3 -m unittest tests.test_runtime.TestStrictLambdaRuntime.test_reject_unknown_action

# Architectural governance checks (run before structural changes)
PYTHONPATH=src python3 -m calosum.harness_checks

# Run a cognitive turn locally
python3 -m calosum.bootstrap.cli run-turn --session-id demo --text "mensagem" --infra-profile persistent

# Interactive chat REPL
python3 -m calosum.bootstrap.cli chat

# Start FastAPI server
python3 -m calosum.bootstrap.api

# Start UI dev server
cd ui && npm run dev

# Docker infrastructure (Qdrant + Jaeger + OTel)
docker compose -f deploy/docker-compose.yml up --build -d
```

## Architecture

Calosum is a neuro-symbolic AI agent framework with dual-hemisphere cognitive architecture. It uses **Ports and Adapters** pattern with **Builder/Factory** for dependency injection.

### Layer Rules (Mechanically Enforced)

The import boundary rules are enforced by `harness_checks.py` via AST analysis. Violations break the build.

- **`shared/`** — Types, Protocols, pure utilities. Zero internal dependencies.
- **`domain/`** — Core cognitive logic. **NEVER imports from `adapters/` or `bootstrap/`**. No external SDK imports (no torch, transformers, httpx, etc.).
- **`adapters/`** — Concrete implementations behind `shared/ports.py` Protocols. All ML libraries (torch, transformers, peft) and external SDKs live here exclusively.
- **`bootstrap/`** — Entry point. The only layer that instantiates adapters and injects them into domain.

### Cognitive Pipeline

```
UserTurn → RightHemisphere (perception/emotion) → Bridge (soft prompts) → LeftHemisphere (reasoning/actions) → StrictLambdaRuntime (safe execution) → Verifier (critique) → repair loop if needed
```

When surprise > 0.6 or ambiguity > 0.8, the orchestrator triggers **group turns**: multiple cognitive variants (conservative/exploratory/creative) evaluated by `GEAReflectionController`, which also adjusts bridge parameters (neuroplasticity).

### Key Contracts

All injectable components are defined as `typing.Protocol` in `shared/ports.py`: `RightHemispherePort`, `CognitiveTokenizerPort`, `LeftHemispherePort`, `MemorySystemPort`, `ActionRuntimePort`, `VerifierPort`, `TelemetryBusPort`, `ReflectionControllerPort`, `BridgeStateStorePort`.

### Fallback Design

Every infrastructure component degrades gracefully: Qdrant → JSONL → in-memory, HuggingFace → heuristic JEPA, OpenAI embeddings → lexical deterministic. The bootstrap never hard-fails on missing optional infrastructure.

## Working Conventions

- Changes touching >1 subsystem require an execution plan in `docs/exec-plans/active/` (see `docs/PLANS.md` for template). Move to `completed/` when done.
- New external integrations must go behind a Protocol in `shared/ports.py`.
- Module size limit: <400 lines per file (enforced by harness).
- Tech debt goes in `docs/exec-plans/tech-debt-tracker.md`.
- All ML/tensor code must stay in `adapters/`. Domain uses only native Python types and shared dataclasses.
- The project uses Portuguese for documentation and semantic keywords in domain logic (emotion labels, salience keywords, plan steps).

## Infrastructure Profiles

Configured via `CALOSUM_INFRA_PROFILE` env var:
- **ephemeral** — RAM-only, for tests
- **persistent** — `.calosum-runtime/` JSONL files, for local dev
- **docker** — Qdrant + OTLP, for production-like environments

Key env vars: `CALOSUM_LEFT_ENDPOINT`, `CALOSUM_LEFT_API_KEY`, `CALOSUM_LEFT_MODEL`, `CALOSUM_VECTORDB_URL`, `CALOSUM_EMBEDDING_ENDPOINT`.
