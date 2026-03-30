# Plano de Execucao: Right Hemisphere Reality Upgrade

## Purpose
Evoluir o Hemisferio Direito de um comportamento majoritariamente heuristico para um subsistema de percepcao com sinais semanticos calibrados, metricas reproduziveis e impacto real na decisao do agente, mantendo execucao local em laptop comum.

## Scope
1. Baseline e observabilidade do Hemisferio Direito
- Definir benchmark local minimo para percepcao: surpresa, salience, estabilidade por seed e impacto no roteamento de turno simples vs group turn.
- Expor no telemetry payload campos de confianca/cobertura do backend de percepcao para diferenciar modo heuristico de modo embedding.

2. Pipeline de percepcao mais realista sem quebrar local-first
- Consolidar adapter de percepcao por embeddings como caminho padrao quando `sentence-transformers` estiver disponivel.
- Enriquecer extracao afetiva com classificador textual leve (multilingual) e fallback deterministico quando dependencia faltar.
- Padronizar contrato do `RightHemisphereState` com provenance explicita: `backend`, `model_name`, `confidence`, `degraded_reason`.

3. Surprise e salience calibrados por memoria
- Revisar calculo de surpresa para combinar distancia vetorial + novidade semantica + recencia de episodios.
- Introduzir calibracao de salience por janela movel da sessao para reduzir picos falsos.
- Adicionar limites e testes de regressao para evitar drift numerico.

4. Fechamento de loop com hemisferio esquerdo e runtime
- Alimentar o hemisferio direito com feedback de falhas/sucessos de execucao para ajustar priorizacao de sinais em turnos futuros.
- Persistir sinais operacionais minimos no workspace para permitir ajuste de percepcao na sessao.

5. Adaptacao continua controlada (sem fine-tuning pesado)
- Implementar atualizacao incremental de parametros de tokenizacao afetiva e thresholds (micro-ajustes) baseada em reflexao.
- Proibir auto-aplicacao de mudancas de topologia/modelo; somente ajustes parametricos pequenos e auditaveis.

6. Avaliacao objetiva orientada a produto
- Criar suite de avaliacao local com:
  - acerto afetivo aproximado (dataset pequeno curado interno),
  - estabilidade de surpresa/salience,
  - impacto em qualidade percebida (empathy_priority util vs ruido),
  - latencia e memoria por perfil (`ephemeral`, `persistent`, `docker`).

## Validation
- `PYTHONPATH=src python3 -m calosum.harness_checks`
- `PYTHONPATH=src python3 -m unittest discover -s tests -t .`
- Benchmark de percepcao salvo em `docs/reports/` com comparativo heuristico vs embedding.
- API/UI exibem backend real do hemisferio direito e estado de degradacao.
- Nao ha quebra de fronteiras de arquitetura (`domain` sem SDK externo; integracoes em `adapters`).

## Implementation Backlog

### Sprint 0: Baseline e Contratos de Telemetria (1-2 dias)
Arquivos alvo:
- `src/calosum/adapters/active_inference.py`
- `src/calosum/adapters/right_hemisphere_hf.py`
- `src/calosum/domain/right_hemisphere.py`
- `src/calosum/bootstrap/factory.py`
- `tests/test_active_inference.py`
- `tests/test_factory.py`
- `tests/test_api.py`

Entregaveis:
- Telemetria padrao do hemisferio direito com chaves estaveis:
  - `right_backend`, `right_model_name`, `right_mode` (`heuristic` ou `embedding`), `degraded_reason`.
- Paridade de telemetria entre adapter heuristico e adapter HF.
- Exposicao consistente de backend/modelo no snapshot de capacidades e dashboard.

Criterio de pronto:
- Testes de factory e API validam campos de backend/modelo/degradacao.

### Sprint 1: Percepcao Afetiva Menos Heuristica (2-4 dias)
Arquivos alvo:
- `src/calosum/adapters/right_hemisphere_hf.py`
- `src/calosum/domain/right_hemisphere.py`
- `src/calosum/shared/types.py`
- `tests/test_right_hemisphere_hf.py`
- `tests/test_pipeline.py`

Entregaveis:
- Extracao afetiva hibrida melhor calibrada (keyword + similaridade vetorial com thresholds por label).
- Confidence calibrada por cobertura de sinais (texto/sinais multimodais) e qualidade do match afetivo.
- Contrato explicito de provenance no `RightHemisphereState.telemetry`.

Criterio de pronto:
- Casos afetivos em PT-BR com reducao de falsos positivos em labels/salience.

### Sprint 2: Surprise e Salience Calibrados por Memoria (2-4 dias)
Arquivos alvo:
- `src/calosum/adapters/active_inference.py`
- `src/calosum/domain/right_hemisphere.py`
- `src/calosum/domain/memory.py`
- `tests/test_active_inference.py`
- `tests/test_memory.py`
- `tests/test_reflection.py`

Entregaveis:
- Surprise combinando distancia latente + novidade semantica + recencia.
- Janela movel de salience por sessao com clamp para reduzir oscilacao.
- Tuning dos limiares que disparam group turn, com base em distribuicao observada.

Criterio de pronto:
- Testes mostram ordenacao estavel: caso novel > caso familiar em surprise.
- Menor taxa de branch falso em prompts neutros repetitivos.

### Sprint 3: Loop Bidirecional com Runtime (2-3 dias)
Arquivos alvo:
- `src/calosum/domain/agent_execution.py`
- `src/calosum/domain/workspace.py`
- `src/calosum/domain/orchestrator.py`
- `tests/test_api.py`
- `tests/test_runtime.py`

Entregaveis:
- Workspace registra sinais de sucesso/falha util para o hemisferio direito.
- Adapter de percepcao incorpora feedback operacional da sessao para priorizacao de contexto.
- Trilhas de auditoria no dashboard para "System 2 corrected System 1".

Criterio de pronto:
- Estado de sessao evidencia a realimentacao e efeito em turnos subsequentes.

### Sprint 4: Benchmark e Hardening (2-3 dias)
Arquivos alvo:
- `tests/` (suite dedicada de benchmark local)
- `docs/reports/`
- `README.md`
- `docs/QUALITY_SCORE.md`

Entregaveis:
- Benchmark reproducivel heuristico vs embedding com dataset pequeno curado.
- Relatorio de latencia/memoria por perfil de infraestrutura.
- Atualizacao de score de qualidade para o eixo "Percepcao Hemisphere Right".

Criterio de pronto:
- Relatorio publicado e usado como gate para evolucoes de topologia/modelo.

## Sequencia de Execucao Recomendada
1. Sprint 0
2. Sprint 1
3. Sprint 2
4. Sprint 3
5. Sprint 4

## Riscos e Mitigacoes
- Risco: aumento de latencia com stacks HF locais.
  Mitigacao: fallback heuristico transparente + limites de timeout + cache de embeddings.
- Risco: calibracao agressiva elevar branch rate e custo de group turns.
  Mitigacao: testes de regressao com cenarios neutros e guardrails de threshold.
- Risco: drift de comportamento entre adapters.
  Mitigacao: padronizar telemetria e testes de contrato compartilhados.

## Progress
- [x] Passo 1: Baseline e observabilidade do hemisferio direito
- [x] Passo 2: Pipeline de percepcao textual-afetiva com provenance
- [x] Passo 3: Calibracao de surprise/salience por memoria
- [x] Passo 4: Loop bidirecional com feedback do runtime
- [x] Passo 5: Adaptacao continua controlada
- [x] Passo 6: Benchmark final e hardening de regressao

## Decision Log
- 2026-03-30: Priorizar evolucao incremental e mensuravel do hemisferio direito em vez de migracao imediata para V-JEPA/video, por compatibilidade com local-first e foco textual atual do produto.
- 2026-03-30: Tratar V-JEPA/M3-JEPA como trilha de pesquisa posterior, atras de adapter e feature flag, sem acoplamento no core.
- 2026-03-30: Organizar execucao em 5 sprints curtos com criterios de pronto testaveis por arquivo para acelerar entrega e reduzir risco de regressao.
- 2026-03-30: Sprint 0 concluida com contrato estavel de telemetria do hemisferio direito no runtime/dashboard e cobertura de testes de API/factory/active inference.
- 2026-03-30: Sprint 1 concluida com thresholds afetivos por label, confidence dinamica por evidencia e teste dedicado para reduzir falso positivo em texto neutro PT-BR.
- 2026-03-30: Sprint 2 concluida com `free_energy_novelty` no active inference e calibracao temporal de salience por sessao para reduzir picos falsos.
- 2026-03-30: Sprint 3 concluida com feedback operacional entre turnos (`previous_runtime_feedback`), bias perceptivo no hemisferio direito e flag auditavel de override cognitivo no dashboard.
- 2026-03-30: Sprint 4 concluida com benchmark local reproduzivel, artefato JSON/relatorio em `docs/reports/` e atualizacao do `QUALITY_SCORE.md` para o eixo de percepcao do hemisferio direito.
- 2026-03-30: Sprint 5 concluida com guardrails de diretivas no orquestrador: bloqueio de TOPOLOGY/ARCHITECTURE, tuning paramétrico pequeno e auditável para `right_hemisphere`, e evidência de regressão via testes de awareness/harness.

## Final Summary
- Ciclo encerrado em 2026-03-30 com os passos 1-6 concluídos.
- Entregas principais: telemetria estável do hemisfério direito, calibração afetiva/temporal, componente de novidade em active inference, loop bidirecional com feedback de runtime, guardrails de adaptação contínua (sem auto-troca de topologia), benchmark local e atualização de qualidade.
- Validação de fechamento: `PYTHONPATH=src python3 -m calosum.harness_checks` e `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t .` passando no estado final.
