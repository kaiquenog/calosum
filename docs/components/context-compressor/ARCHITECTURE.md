# Context Compressor (alias: CognitiveTokenizer)

## Purpose
Converter estado latente do hemisferio direito em pacote simbolico controlavel para o hemisferio esquerdo, aplicando Information Bottleneck.

## Boundaries
- Camada: `domain/cognition`.
- Implementacao canonica: `ContextCompressor` em `src/calosum/domain/cognition/bridge.py`.
- Alias de compatibilidade (Sprint 0): `CognitiveTokenizer`.

## Inputs / Outputs
- Input: `RightHemisphereState`.
- Output: `CognitiveBridgePacket` com `soft_prompts`, `control_signal`, `salience` e metadados do bridge.

## Runtime Notes
- Parametros de comportamento sao adaptados pelo loop de neuroplasticidade apos group turns.
- Persistencia de estado fica fora do domain (adapters/bootstrap).
