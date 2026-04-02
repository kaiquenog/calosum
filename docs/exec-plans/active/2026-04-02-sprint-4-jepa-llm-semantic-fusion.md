# Sprint 4 JEPA LLM Semantic Fusion

## Purpose
Implementar fusao semantica genuina JEPA+LLM: JEPA prediz embedding alvo, LLM gera multiplos candidatos e um seletor escolhe o mais alinhado semanticamente.

## Scope
- Adicionar modo multi-sample no hemisferio esquerdo com `n_candidates` configuravel.
- Implementar `SemanticFusionSelector` com fallback por incerteza.
- Integrar trigger por `jepa_uncertainty < 0.5`.
- Suportar controle de feature por `CALOSUM_FUSION_ENABLED`.
- Suportar condicao de teste A/B via selecao `guided` (A) e `random` (B).
- Instrumentar telemetria de fusao no `left_result.telemetry`.
- Publicar protocolo e resultado de benchmark em `docs/benchmarks/fusion_ab_test.md`.

## Validation
- `PYTHONPATH=src ./.venv/bin/python3 -m calosum.harness_checks`
- `PYTHONPATH=src ./.venv/bin/python3 -m unittest tests.adapters.llm.test_llm_adapter tests.adapters.llm.test_llm_fusion tests.bootstrap.test_fusion_resolver tests.domain.metacognition.test_reflection`
- Verificar que profile `ephemeral` nao ativa fusao por default.
- Verificar que `CALOSUM_FUSION_ENABLED=false` desativa sem quebrar fluxo.

## Progress
- [x] Desenho de arquitetura para inserir fusao sem aumentar acoplamento do domain core.
- [x] Implementacao de `MultiSampleFusionLeftHemisphereAdapter` e `SemanticFusionSelector`.
- [x] Integração no resolver de backend com flags de ambiente.
- [x] Ajuste no adapter LLM para usar `target_temperature` do bridge packet.
- [x] Testes unitarios iniciais da fusao e resolver.
- [ ] Execucao do A/B test formal com LLM-as-judge e analise estatistica p<0.05.

## Decision Log
- 2026-04-02: Fusao foi encapsulada no adapter de LLM para evitar alterar `orchestrator.py` e `agent_execution.py`, que ja estao no limite de tamanho de modulo.
- 2026-04-02: Trigger de fusao usa `jepa_uncertainty` propagada em `bridge_packet.control.annotations` para manter contratos existentes.
- 2026-04-02: `ephemeral` desabilita fusao por default para evitar custo extra de chamadas LLM; perfis persistentes/docker habilitam por default com override por env.
