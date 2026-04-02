# Cognitive Variant Selector (alias: GEAReflectionController)

## Purpose
Avaliar candidatos cognitivos (group turn), selecionar variante vencedora e produzir outcome para aprendizagem adaptativa.

## Boundaries
- Camada: `domain/metacognition`.
- Implementacao canonica: `CognitiveVariantSelector` em `src/calosum/domain/metacognition/metacognition.py`.
- Alias de compatibilidade (Sprint 0): `GEAReflectionController`.

## Inputs / Outputs
- Input: lista de `CognitiveCandidate` + estado de contexto da sessao.
- Output: `ReflectionOutcome` (vencedor, scores, ajustes recomendados).

## Runtime Notes
- Pode receber priors de experiencia via adapters (`ExperienceAware*` / preferencia aprendida).
- Nao executa acoplamento direto com infra externa; usa ports.
