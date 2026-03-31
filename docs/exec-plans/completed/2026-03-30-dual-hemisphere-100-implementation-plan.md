# 2026-03-30 - Dual Hemisphere 100 Implementation Plan

## Purpose

Elevar o Calosum do estado atual para o aspiracional dual-hemisphere 2026 com implementacao real de:
- Hemisferio direito preditivo (V-JEPA 2.1 / VL-JEPA / AC) com backend local.
- Hemisferio esquerdo com RLM oficial e fallback para adapter atual.
- Corpus caloso com fusao cross-attention e bottleneck governado.
- GEA com experience sharing persistente e reflexao evolutiva real.
- Operacao local-first com backend opcional `jepa-rs` (Rust + Burn).

Objetivos de saida (Definition of Done do programa):
1. Novo pipeline habilitavel por feature flags sem quebrar o fluxo atual.
2. `Ports and Adapters` preservado (dominio sem acoplamento direto ao stack infra).
3. `harness_checks` e suite de testes verdes.
4. Documentacao operacional atualizada para bootstrap, env vars, fallback e observabilidade.

## Scope

### Baseline confirmado no codigo (inspecao previa obrigatoria)

1. Right hemisphere atual e heuristico/embedding, nao world model preditivo temporal:
- `src/calosum/domain/right_hemisphere.py:35-218`
- `src/calosum/adapters/right_hemisphere_hf.py:53-319`
- `src/calosum/bootstrap/factory.py:109-147`

2. Active inference existe, mas surpresa e derivada de distancia de embeddings + priors:
- `src/calosum/adapters/active_inference.py:130-207`

3. Bridge atual usa heuristica ou MLP simples, sem cross-attention treinavel:
- `src/calosum/domain/bridge.py:40-170`

4. Left hemisphere atual e LLM structured output; sem runtime RLM oficial:
- `src/calosum/domain/left_hemisphere.py:24-325`
- `src/calosum/adapters/llm_qwen.py:51-367`

5. GEA atual e heuristico (score + UCB), sem archive de experiencia coletiva robusta:
- `src/calosum/domain/metacognition.py:110-391`
- `src/calosum/adapters/latent_exchange.py:9-41`
- `src/calosum/domain/evolution.py:21-236`

6. Restricoes estruturais relevantes para planejamento:
- Limite de modulo = 400 linhas (`src/calosum/harness_checks.py:207`)
- Modulos quase no limite:
  - `src/calosum/domain/orchestrator.py` = 400
  - `src/calosum/bootstrap/factory.py` = 390
  - `src/calosum/domain/metacognition.py` = 391
  - `src/calosum/shared/types.py` = 391
- Regras de fronteira AST/import exigem registrar todo modulo novo em `MODULE_RULES`:
  - `src/calosum/harness_checks.py:46-205`

### In Scope

1. Novos adapters e wiring em factory/settings para backends 2026.
2. Atualizacao da matematica de surprise/efe sem quebrar contrato `RightHemisphereState`.
3. Integracao RLM com saida compatibilizada para `LeftHemisphereResult`.
4. Evolucao da reflexao GEA para sharing persistente.
5. Testes unitarios/integracao e docs operacionais.

### Out of Scope (neste plano)

1. Treinamento foundation-scale de JEPA no repositorio.
2. Dependencia obrigatoria de GPU para funcionamento baseline.
3. Mudanca do protocolo de API publica do Calosum (resposta externa continua compativel).

### Arquitetura alvo (resumo de implementacao)

1. Right: `ActiveInferenceRightHemisphereAdapter(base_adapter=VJepa21|VLJepa|JepaRs|HF|JEPA)`
2. Bridge: tokenizacao + fusao por estrategia injetavel (heuristica/projection/cross-attention).
3. Left: `RlmLeftHemisphereAdapter` com fallback para `QwenLeftHemisphereAdapter` e failover atual.
4. Reflection: `GEAReflectionController` apoiado por `ExperienceStorePort` persistente.

---

### Sprint 0 - Preflight arquitetural e guardrails (bloqueante)

Objetivo:
- Preparar terreno para evolucao sem quebrar limites de modulo, fronteiras e compatibilidade.

Arquivos alvo:
- Alterar:
  - `src/calosum/harness_checks.py`
  - `src/calosum/bootstrap/factory.py`
  - `src/calosum/bootstrap/settings.py`
- Criar:
  - `src/calosum/adapters/right_hemisphere_vjepa21.py`
  - `src/calosum/adapters/right_hemisphere_vljepa.py`
  - `src/calosum/adapters/right_hemisphere_jepars.py`
  - `src/calosum/adapters/left_hemisphere_rlm.py`
  - `src/calosum/adapters/gea_experience_store.py`

Passos:
1. Criar adapters inicialmente como stubs funcionais com fallback seguro.
2. Registrar modulos novos em `MODULE_RULES` imediatamente para evitar drift AST.
3. Adicionar feature flags em `InfrastructureSettings` sem ativar por default.
4. Atualizar factory para reconhecer novos backends e manter default atual.

Criterios de aceite:
- Build sem regressao com backend default atual.
- Sem novos erros em `harness_checks`.
- Nenhum modulo ultrapassa 400 linhas.

Risco principal:
- Estouro de linhas em `factory.py` e `settings.py`.

Mitigacao:
- Extrair funcao de resolucao de backends para modulo auxiliar se necessario.

Rollback:
- Reverter somente wiring de flags e manter stubs isolados sem uso.

---

### Sprint 1 - Contratos e modelos de dados para 2026 (Ports and Adapters)

Objetivo:
- Introduzir contratos minimos para preditor latente, bridge fusion e experience sharing sem quebrar componentes atuais.

Arquivos alvo:
- Alterar:
  - `src/calosum/shared/ports.py`
  - `src/calosum/shared/types.py` (apenas mudancas pequenas para nao estourar limite)
  - `src/calosum/__init__.py`

Passos detalhados:
1. Adicionar novos Ports:
- `BridgeFusionPort` (fuse right/left latents -> fused vector + metadata).
- `ExperienceStorePort` (append/query/share experiences por sessao/contexto).

2. Evitar crescimento excessivo em `shared/types.py`:
- Nao criar dataclasses extensas nesta fase.
- Reutilizar `telemetry` e `world_hypotheses` para novos campos ate extracao futura.

3. Manter compatibilidade binaria:
- Nenhum metodo existente deve mudar assinatura obrigatoria.
- Novos recursos entram como opcionais e feature-flagged.

Criterios de aceite:
- Componentes antigos continuam operando sem conhecer novos ports.
- Testes atuais de pipeline/factory/reflection continuam verdes.

Risco principal:
- `shared/types.py` (391 linhas) passar do limite.

Mitigacao:
- Se necessario, abrir sprint tecnico paralelo para extrair `shared/types_cognitive.py` e atualizar imports + `MODULE_RULES`.

Rollback:
- Remover apenas novos ports e manter adapters desacoplados via typing local.

---

### Sprint 2 - Hemisferio direito V-JEPA 2.1 (local-first)

Objetivo:
- Substituir backend de percepcao por preditor latente temporal real (ONNX local).

Arquivos alvo:
- Criar:
  - `src/calosum/adapters/right_hemisphere_vjepa21.py`
  - `tests/test_right_hemisphere_vjepa21.py`
- Alterar:
  - `src/calosum/bootstrap/factory.py`
  - `src/calosum/bootstrap/settings.py`
  - `README.md`
  - `docs/INFRASTRUCTURE.md`

Passos detalhados:
1. Implementar `VJepa21RightHemisphereAdapter` com:
- loader local de encoder/predictor.
- suporte a hint de acao (`action_conditioned`).
- inferencia robusta para `UserTurn.signals` multimodais quando disponivel.

2. Integrar com wrapper de active inference existente:
- manter `ActiveInferenceRightHemisphereAdapter` como camada superior.

3. Adicionar env vars:
- `CALOSUM_RIGHT_BACKEND`
- `CALOSUM_RIGHT_MODEL_PATH`
- `CALOSUM_RIGHT_ACTION_CONDITIONED`
- `CALOSUM_RIGHT_HORIZON`

4. Atualizar factory para rotear por backend selecionado.

Criterios de aceite:
- Quando `CALOSUM_RIGHT_BACKEND=vjepa21`, builder monta stack novo.
- Fallback para backend atual em erro de carga (degraded_reason explicito).
- `surprise_score` continua no intervalo [0,1].

Risco principal:
- Dependencias opcionais ausentes no ambiente local.

Mitigacao:
- Fail-open para backend atual com telemetria clara.

Rollback:
- Reconfigurar `CALOSUM_RIGHT_BACKEND` para backend legado.

---

### Sprint 3 - Backend VL-JEPA com features hierarquicas

Objetivo:
- Adicionar variante multimodal mais rica para cenarios texto+visao.

Arquivos alvo:
- Criar:
  - `src/calosum/adapters/right_hemisphere_vljepa.py`
  - `tests/test_right_hemisphere_vljepa.py`
- Alterar:
  - `src/calosum/bootstrap/factory.py`
  - `README.md`

Passos:
1. Implementar adaptador com extracao hierarquica de features latentes.
2. Projetar sinais hierarquicos para `world_hypotheses` (semantic_density, visual_richness, ambiguity).
3. Garantir caminho local-first (sem chamada remota obrigatoria).

Criterios de aceite:
- Backend VL-JEPA selecionavel por env var.
- Sem regressao para entradas somente texto.

Risco:
- Latentes de dimensoes diferentes quebrarem bridge.

Mitigacao:
- Normalizador de dimensao no bridge (pad/truncate/projection).

Rollback:
- fallback automatico para `vjepa21` ou backend legado.

---

### Sprint 4 - Backend opcional jepa-rs (Rust + Burn)

Objetivo:
- Entregar opcao de inferencia JEPA em runtime Rust local para robustez/performance.

Arquivos alvo:
- Criar:
  - `src/calosum/adapters/right_hemisphere_jepars.py`
  - `tests/test_right_hemisphere_jepars.py`
  - `docs/references/jepa-rs-integration.md`
- Alterar:
  - `src/calosum/bootstrap/factory.py`
  - `src/calosum/bootstrap/settings.py`
  - `README.md`

Passos:
1. Definir contrato IPC local (stdin/stdout JSON ou socket unix).
2. Adapter Python chama runtime Rust e normaliza retorno.
3. Implementar healthcheck no bootstrap para marcar backend `healthy/degraded`.

Criterios de aceite:
- `CALOSUM_RIGHT_BACKEND=jepars` funcional com fallback transparente.
- Telemetria identifica backend real ativo.

Risco:
- Friccao de deploy local por toolchain Rust.

Mitigacao:
- manter opcional; documentar bootstrap minimo e fallback automatico.

Rollback:
- desabilitar backend `jepars` por env.

---

### Sprint 5 - Hemisferio esquerdo com RLM oficial

Objetivo:
- Introduzir recursao linguistica oficial RLM mantendo fronteira `LeftHemispherePort`.

Arquivos alvo:
- Criar:
  - `src/calosum/adapters/left_hemisphere_rlm.py`
  - `tests/test_left_hemisphere_rlm.py`
- Alterar:
  - `src/calosum/bootstrap/factory.py`
  - `src/calosum/bootstrap/settings.py`
  - `src/calosum/domain/agent_execution.py`
  - `README.md`

Passos detalhados:
1. Implementar wrapper RLM com metodos `reason/areason/repair/arepair`.
2. Converter saida recursiva para:
- `response_text`
- `TypedLambdaProgram`
- `PrimitiveAction[]`
- `reasoning_summary`

3. Enriquecer loop de repair:
- preservar feedback cumulativo (`agent_execution.py:145-365`) para recursion guidance.

4. Novas env vars:
- `CALOSUM_LEFT_BACKEND=rlm|qwen`
- `CALOSUM_LEFT_RLM_PATH`
- `CALOSUM_LEFT_RLM_MAX_DEPTH`

Criterios de aceite:
- Backend RLM selecionavel e observavel.
- Fallback para Qwen quando RLM indisponivel.
- Runtime DSL continua recebendo acoes tipificadas validas.

Risco:
- Saidas RLM nao aderirem ao schema esperado.

Mitigacao:
- camada robusta de normalizacao + fallback para resposta segura.

Rollback:
- alternar `CALOSUM_LEFT_BACKEND=qwen`.

---

### Sprint 6 - Corpus caloso com cross-attention + bottleneck

Objetivo:
- Migrar de projection/heuristica para fusao cross-attention com gating.

Arquivos alvo:
- Criar:
  - `src/calosum/adapters/bridge_cross_attention.py`
  - `tests/test_bridge_cross_attention.py`
- Alterar:
  - `src/calosum/domain/bridge.py`
  - `src/calosum/shared/ports.py`
  - `src/calosum/bootstrap/factory.py`

Passos:
1. Introduzir estrategia de fusao injetavel por port (sem acoplamento de dominio a torch).
2. Bridge passa a:
- usar fusao quando backend habilitado.
- manter heuristica atual como fallback.
3. Registrar metadados de fusao em `bridge_metadata` para auditoria.

Criterios de aceite:
- `CALOSUM_BRIDGE_BACKEND=cross_attention` funciona.
- Fallback para heuristica sem quebra.

Risco:
- Domain bridge crescer demais com estrategia multipla.

Mitigacao:
- extrair helper functions para modulo auxiliar mantendo import boundaries.

Rollback:
- `CALOSUM_BRIDGE_BACKEND=heuristic`.

---

### Sprint 7 - Active Inference refinado (VFE/EFE multi-horizonte)

Objetivo:
- Atualizar calculo de surpresa para erro preditivo temporal + incerteza.

Arquivos alvo:
- Alterar:
  - `src/calosum/adapters/active_inference.py`
  - `tests/test_active_inference.py`
- Criar:
  - `tests/test_free_energy_refined.py`

Passos:
1. Incluir decomposicao explicita em telemetria:
- risk
- ambiguity
- epistemic_value
- novelty_density

2. Adicionar suporte multi-horizonte usando preditos do backend JEPA quando presentes.
3. Manter compatibilidade com fallback atual quando apenas vetor simples estiver disponivel.

Criterios de aceite:
- Telemetria consistente para old/new backends.
- `surprise_score` estavel e calibrado [0,1].

Risco:
- regressao de escala de surpresa impactando branching.

Mitigacao:
- testes de regressao + calibracao por threshold config.

Rollback:
- flag para engine anterior (`active_inference_legacy=true`).

---

### Sprint 8 - GEA real com experience sharing persistente

Objetivo:
- Evoluir reflexao para dinamica coletiva com memoria compartilhada entre variantes/agentes.

Arquivos alvo:
- Criar:
  - `src/calosum/adapters/gea_experience_store.py`
  - `tests/test_gea_experience_sharing.py`
- Alterar:
  - `src/calosum/domain/metacognition.py`
  - `src/calosum/domain/orchestrator.py`
  - `src/calosum/bootstrap/factory.py`
  - `src/calosum/shared/ports.py`

Passos:
1. Introduzir store persistente (jsonl/sqlite) para experiencias por contexto.
2. Reflection usa experiencias historicas na funcao de score/selecion.
3. Integrar com `latent_exchange` sem substituir compatibilidade atual.

Observacao critica de engenharia:
- `orchestrator.py` ja esta com 400 linhas; qualquer nova logica exige extracao antes:
  - Criar modulo auxiliar (ex.: `domain/orchestrator_group.py`) para fluxo de group turn e sharing.

Criterios de aceite:
- Score de reflexao considera experiencia historica.
- Persistencia de experiencia habilitavel/desabilitavel por env.

Risco:
- crescimento excessivo de `metacognition.py` (391 linhas).

Mitigacao:
- extrair scorer e reward engine para modulo dedicado.

Rollback:
- desabilitar sharing (`CALOSUM_GEA_SHARING_ENABLED=false`).

---

### Sprint 9 - Wiring final, docs, benchmark e operacao

Objetivo:
- Fechar ciclo de produto: bootstrap, docs, benchmark e checklist de producao.

Arquivos alvo:
- Alterar:
  - `README.md`
  - `docs/ARCHITECTURE.md`
  - `docs/INFRASTRUCTURE.md`
  - `docs/QUALITY_SCORE.md`
  - `docs/RELIABILITY.md`
  - `src/calosum/bootstrap/settings.py`
  - `src/calosum/bootstrap/factory.py`
- Criar:
  - `docs/reports/2026-xx-xx-dual-hemisphere-benchmark.md`
  - `tests/test_factory_backends_2026.py`
  - `tests/test_pipeline_dual_hemisphere_e2e.py`

Passos:
1. Documentar matriz de backends e fallback.
2. Publicar env vars novas e defaults recomendados.
3. Rodar benchmark comparando legacy vs 2026 backends em casos controlados.
4. Atualizar scorecard em docs.

Criterios de aceite:
- Docs refletem comportamento real do codigo.
- Factory cobre combinacoes principais de backend.

Risco:
- divergencia entre docs e implementacao.

Mitigacao:
- testes de factory e snapshots de `builder.describe()` para cada backend.

Rollback:
- manter docs legacy em branch de contingencia, se necessario.

---

### Backlog tecnico transversal (obrigatorio em paralelo)

1. Controle de tamanho de modulo (MAX 400):
- Se qualquer modulo ultrapassar limite, extrair imediatamente.
- Prioritarios para extracao:
  - `domain/orchestrator.py`
  - `domain/metacognition.py`
  - `bootstrap/factory.py`
  - `shared/types.py`

2. Governanca AST:
- Toda criacao de modulo exige atualizacao imediata de `MODULE_RULES`.

3. Compatibilidade gradual:
- Nenhum sprint deve remover backend legacy antes de equivalente novo estar validado.

4. Telemetria canonica:
- Manter chaves estaveis:
  - `right_backend`
  - `right_model_name`
  - `right_mode`
  - `degraded_reason`
  - `surprise_backend`

---

## Validation

### Gate A - Arquitetura e guardrails

Executar apos cada sprint estrutural:

```bash
PYTHONPATH=src python3 -m calosum.harness_checks
```

Pass criteria:
1. Sem `missing_module_rule`.
2. Sem `forbidden_internal_import`.
3. Sem `module_too_large`.

### Gate B - Testes unitarios e integracao

Executar suite completa antes de fechar cada sprint de codigo:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -t .
```

E rodar testes focados da frente:

```bash
PYTHONPATH=src python3 -m unittest \
  tests.test_factory \
  tests.test_active_inference \
  tests.test_reflection \
  tests.test_pipeline
```

### Gate C - Validacao funcional por backend

Casos minimos por combinacao:
1. `RIGHT=vjepa21`, `LEFT=rlm`, `BRIDGE=cross_attention`, `GEA_SHARING=true`.
2. `RIGHT=jepars`, `LEFT=qwen`, `BRIDGE=heuristic`, `GEA_SHARING=false`.
3. Falha forcada em right/left para verificar fallback e degradacao limpa.

### Gate D - Regressao de telemetria

Verificar por asserts/snapshots:
1. `builder.describe()` mostra backend ativo correto.
2. Eventos de dashboard continuam com estrutura esperada (`felt/thought/decision/execution/reflection`).
3. `surprise_score` e `confidence` permanecem dentro de faixas validas.

### Gate E - Benchmark tecnico

Minimo:
1. Medir latencia, retries, surprise medio, taxa de sucesso de acoes.
2. Comparar baseline atual vs stack 2026 em conjunto fixo de prompts.
3. Registrar resultado em `docs/reports/` com data.

## Progress

- [x] Inspecao tecnica do estado atual (codigo e docs centrais).
- [x] Criacao do plano detalhado em `docs/exec-plans/active/`.
- [x] Sprint 0 concluido — Stubs e feature flags sem quebrar baseline.
- [x] Sprint 1 concluido — Ports `BridgeFusionPort`, `ExperienceStorePort` em `shared/ports.py`.
- [x] Sprint 2 concluido — `VJepa21RightHemisphereAdapter` funcional com fallback.
- [x] Sprint 3 concluido — `VLJepaRightHemisphereAdapter` multimodal.
- [x] Sprint 4 concluido — `JepaRsRightHemisphereAdapter` (Rust/Burn backend).
- [x] Sprint 5 concluido — `RlmLeftHemisphereAdapter` com fallback para Qwen.
- [x] Sprint 6 concluido — `CrossAttentionBridgeAdapter` com `BridgeFusionPort`.
- [x] Sprint 7 concluido — Active Inference refinado com EFE multi-horizonte e novelty density.
- [x] Sprint 8 concluido — GEA experience sharing persistente (SQLite + Redis).
- [x] Sprint 9 concluido — Wiring final, docs operacionais, benchmark de hemisferio direito.
- [x] Harness checks final verde (2026-03-31).
- [x] Suite de testes final verde — 114 testes passando (2026-03-31).
- [x] Plano movido para `docs/exec-plans/completed/` com resumo final.

## Resumo Final (2026-03-31)

Todos os 10 sprints do programa dual-hemisphere 2026 foram concluídos. O Calosum evoluiu de um
framework heurístico para uma arquitetura neuro-simbólica dual-hemisphere completa com:

- 4 backends selecionáveis para o hemisfério direito (HF/V-JEPA 2.1/VL-JEPA/JEPA-rs).
- Backend RLM recursivo para o hemisfério esquerdo com fallback para Qwen.
- Cross-attention aprendida no corpus caloso com BridgeFusionPort.
- Active Inference com EFE multi-horizonte e decomposição de surpresa.
- GEA com experience sharing persistente (SQLite) e distribuído (Redis).
- DSPy e LoRA para ciclos noturnos de auto-otimização.
- Contract wrappers por hemisferio garantindo contratos estáveis em runtime.
- Backend resolver centralizado mantendo fábrica desacoplada de decisões de routing.
- 114 testes automatizados cobrindo todas as camadas.
- Harness verde com verificação AST mecânica de fronteiras e tamanho de módulos.

Debt residual documentado em `docs/exec-plans/tech-debt-tracker.md`.

## Decision Log

1. 2026-03-30 - O rollout sera incremental por feature flags, nao big-bang.
Razao: factory atual ja suporta fallback e precisa manter estabilidade operacional (`src/calosum/bootstrap/factory.py:85-147`).

2. 2026-03-30 - Active inference permanece como camada wrapper no right hemisphere.
Razao: arquitetura atual ja abstrai bem esse papel e reduz risco de regressao (`src/calosum/adapters/active_inference.py:24-98`).

3. 2026-03-30 - RLM entra como novo `LeftHemispherePort` adapter com fallback para Qwen.
Razao: contrato atual do orchestrator/engine depende de `reason/repair` e feedback loops (`src/calosum/domain/agent_execution.py:40-287`).

4. 2026-03-30 - Cross-attention sera injetado por port/adapter, evitando acoplamento no dominio.
Razao: preservar fronteiras `Ports and Adapters` e compatibilidade com harness AST (`src/calosum/harness_checks.py:46-205`).

5. 2026-03-30 - Experience sharing persistente sera introduzido sem remover `InternalLatentExchangeAdapter`.
Razao: permitir rollout progressivo com fallback de baixo risco (`src/calosum/adapters/latent_exchange.py:9-41`).

6. 2026-03-30 - Controle de tamanho de modulo e critico e bloqueante.
Razao: arquivos core estao no limite de 400 linhas (`orchestrator.py`, `metacognition.py`, `factory.py`, `types.py`).

7. 2026-03-30 - Validacao obrigatoria segue AGENTS.md (harness + unittest completo).
Razao: regra de trabalho do repositorio para mudancas estruturais e de codigo.

