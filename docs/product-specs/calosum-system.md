# Calosum — System Product Spec

## Overview

Calosum is a neuro-symbolic AI agent framework featuring a dual-hemisphere cognitive architecture. It combines perception based on embeddings (Right Hemisphere) with LLM-based reasoning (Left Hemisphere), safe action execution, and metacognition inspired by Group-Evolving Agents (GEA).

## Core Properties

- **Dual-Hemisphere**: System 1 (perception/intuition) and System 2 (reasoning/logic) operating in tandem.
- **Ports and Adapters**: All external integrations are behind Protocol interfaces.
- **Active Inference**: Free energy minimization drives attention and branching decisions.
- **Local-First**: Designed for local GPU/CPU execution with graceful degradation.
- **Harness-Governed**: AST-based mechanical enforcement of architectural boundaries.

## Key Components

| Component | Layer | Description |
|---|---|---|
| Right Hemisphere | adapters | Latent perception via V-JEPA 2.1, HF embeddings, or jepa-rs |
| Left Hemisphere | adapters | LLM reasoning via Ollama/Qwen or RLM recursive |
| Bridge (Corpus Callosum) | domain | Cognitive tokenization of latent state for LH consumption |
| Memory System | domain | Dual episodic + semantic memory with sleep consolidation |
| GEA Reflection | domain | Multi-variant metacognition with UCB1 strategy selection |
| Active Inference | adapters | VFE/EFE computation wrapping perception pipeline |
| Runtime | domain | Strict lambda execution with typed action verification |

## Non-Functional Requirements

- Modules must not exceed 400 lines.
- Domain layer must never import external SDKs.
- All changes affecting more than one subsystem require documented plans.
