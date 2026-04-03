# Estabilizacao da Suite de Testes (JEPA, Embeddings, Contratos)

## Purpose
Restaurar a confiabilidade da esteira local corrigindo regressões em fallbacks de embeddings, contrato de wrappers e telemetria/perception status dos adapters de hemisferio.

## Scope
- Corrigir `TextEmbeddingAdapter` para sempre retornar vetores validos (incluindo modo lexical).
- Corrigir adapters `input_perception_heuristic_jepa.py`, `input_perception_trained_jepa.py` e `input_perception_vjepa21.py` para evitar estado cego por indisponibilidade opcional de stack local.
- Ajustar `ContractEnforcedRightHemisphereAdapter` para expor `base_adapter` conforme contrato usado na factory.
- Ajustar backend de telemetria do RLM AST para manter compatibilidade de contrato.
- Validar com `unittest discover` completo.

## Validation
- `PYTHONPATH=src ./.venv/bin/python3 -m unittest discover -s tests -t .` deve finalizar sem erros/falhas.
- `PYTHONPATH=src ./.venv/bin/python3 -m calosum.harness_checks` deve passar, incluindo regras de modulo e fallback ruidoso.
- Telemetria minima esperada mantida: `surprise_source`, `prediction_method`, `jepa_uncertainty`, `right_backend`.

## Progress
- [x] Baseline executado: 11 erros e 9 falhas mapeados.
- [x] Correcao aplicada em adapters de embeddings e hemisferio direito.
- [x] Correcao aplicada no wrapper de contrato (`base_adapter`) e backend RLM.
- [x] Rodar suite completa novamente e registrar resultado (`177` testes, `OK`).
- [x] Rodar harness checks novamente e registrar resultado (`Harness checks passed`).

## Decision Log
- 2026-04-03: Fallback deterministico sem `hashlib` foi adotado para evitar ruido artificial e atender regras de harness.
- 2026-04-03: Em vez de `BLIND` em falta de cache local opcional, adapters passaram a degradar com vetor lexical deterministico para preservar pipeline e telemetria.
- 2026-04-03: `ContractEnforcedRightHemisphereAdapter` passou a expor `base_adapter` explicitamente para manter introspeccao da factory e testes de roteamento.
