# Sprint 1 - JEPA Phase 1 Interface, Heuristic Adapter e Benchmark
## Purpose
Definir o contrato JEPA textual com semantica de predicao latente, integrar um adapter heuristico como backend padrao do hemisferio direito e validar a hipotese com benchmark anotado antes de treinar um JEPA real.

## Scope
- Introduzir contrato `JEPARightHemispherePort` em `shared.models.ports` com:
  - `encode_context(turns) -> ContextEmbedding`
  - `predict_response_embedding(ctx) -> ResponsePrediction`
  - `compute_surprise(ctx, actual_response) -> SurpriseScore`
- Introduzir tipos JEPA em `shared.models.types`:
  - `ContextEmbedding`
  - `ResponsePrediction` (`predicted_embedding`, `uncertainty`, `prediction_method`)
  - `SurpriseScore`
- Implementar `HeuristicJEPAAdapter` em `adapters/hemisphere` usando media ponderada por recencia de embeddings de contexto (384 dims).
- Integrar o adapter como default do right hemisphere no bootstrap resolver.
- Garantir que `surprise_score` venha de `prediction_error` JEPA (predito vs resposta real), preservando essa fonte no wrapper de active inference.
- Aplicar politicas de pipeline:
  - baixa surpresa `<0.3`: fluxo normal
  - media surpresa `0.3-0.6`: registrar telemetria
  - alta surpresa `>0.6`: acionar branching/cognitive variants
  - alta incerteza `>0.7`: ignorar surpresa no branching do turno
- Expandir telemetria com `prediction_method` e `surprise_source`.
- Criar benchmark fase 1 em `docs/benchmarks/jepa_phase1/` com 200 turnos anotados e script de avaliacao:
  - top-1 ranking da resposta boa >= 60%
  - surpresa > 0.5 em respostas fora de contexto >= 70%

## Validation
- `PYTHONPATH=src ./.venv/bin/python3 -m calosum.harness_checks`
- `PYTHONPATH=src ./.venv/bin/python3 -m unittest tests/adapters/hemisphere/test_right_hemisphere_heuristic_jepa.py tests/adapters/perception/test_active_inference.py tests/domain/metacognition/test_reflection.py tests/bootstrap/test_factory.py`
- `PYTHONPATH=src ./.venv/bin/python3 scripts/jepa_phase1_benchmark.py --dataset docs/benchmarks/jepa_phase1/annotated_turns.jsonl --output-json docs/reports/jepa_phase1_benchmark_2026-04-02.json`

## Progress
- [x] Contratos/tipos JEPA adicionados em `shared`.
- [x] `HeuristicJEPAAdapter` implementado com preditor por recencia.
- [x] Pipeline integrado (resolver + active inference + orchestrator thresholds).
- [x] Telemetria expandida com `prediction_method` e `surprise_source`.
- [x] Benchmark fase 1 (dataset + script) criado e executado.
- [x] Testes focados e `harness_checks` passando.

## Decision Log
- 2026-04-02: Fase 1 adota predicao heuristica interpretavel para validar hipotese sem treino JEPA.
- 2026-04-02: `surprise_score` para backend JEPA passa a ser derivado de erro preditivo embedding->embedding em vez de distancia de memoria media.
- 2026-04-02: Para reduzir falsos positivos, `uncertainty > 0.7` neutraliza gating por surpresa no turno corrente.
- 2026-04-02: Benchmark fase 1 em 200 turnos anotados atingiu ranking top-1=1.0 e off-topic surprise rate=1.0.
