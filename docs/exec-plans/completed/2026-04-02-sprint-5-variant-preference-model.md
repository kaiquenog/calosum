# Sprint 5 - Variant Preference Model

## Purpose
Substituir a selecao de variantes cognitivas baseada em pesos arbitrarios por um seletor aprendido com dados reais de group turns, mantendo fallback gracioso e telemetria auditavel.

## Scope
- Coletar dataset de treino por group turn (scores, variante selecionada, rating e contexto).
- Implementar modelo de preferencia com features:
  - surprise_score
  - ambiguity_score
  - intent_type
  - session_length
  - avg_tool_success_rate
  - jepa_uncertainty
- Treinar modelo LightGBM (`n_estimators=50`, `max_depth=4`) com holdout.
- Aplicar politica de selecao com `selected_by`:
  - `learned_model`
  - `rule_based`
  - `legacy`
- Manter heuristicas de fallback ate haver dados suficientes.

## Validation
- Testes unitarios cobrindo coleta, fallback e telemetria de selecao.
- Harness checks passando apos registrar novos modulos/import boundaries.
- Script de treino retornando metadados de amostra e acuracia de holdout.

## Progress
- [x] Implementar store/dataset de treinamento do seletor.
- [x] Implementar treino/predicao do VariantPreferenceModel.
- [x] Integrar seletor ao reflection controller.
- [x] Expor `selected_by` no payload de reflection.
- [x] Atualizar wiring + testes + harness.

## Decision Log
- 2026-04-02: Manter fallback rule-based e legacy para degradacao segura enquanto o modelo ainda nao atinge janela minima de dados.

## Summary
- Sprint concluido em 2026-04-02 e movido para `completed/`.
- Selecao de variantes passa a suportar caminho aprendido com fallback explicito e telemetria `selected_by`.
