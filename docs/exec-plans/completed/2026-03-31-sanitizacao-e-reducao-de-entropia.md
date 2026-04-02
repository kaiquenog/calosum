# 2026-03-31 - Sanitização e Redução de Entropia do Calosum

## Purpose

Reduzir a entropia acumulada no repositório após múltiplos sprints de evolução acelerada.
O projeto está funcional (harness verde, 114 testes passando), mas apresenta divergências entre
documentação e código, regras mortas no harness, e artefatos de diagnóstico misturados com
código de produção. O objetivo é restaurar a legibilidade e a governança de primeira classe.

## Scope

### Diagnóstico de Estado Atual (2026-03-31)

**Harness:** `PYTHONPATH=src .venv/bin/python -m calosum.harness_checks` → **PASSED**
**Testes:** 114 testes, todos passando em ~101s.

---

### Problema 1 — MODULE_RULES com entradas mortas (ghost rules)

`harness_checks.py` registra 3 módulos que não existem no repositório:

| Regra morta | Arquivo esperado | Status |
|---|---|---|
| `final_prod_val` | `src/calosum/final_prod_val.py` | ❌ não existe |
| `verify_v3` | `src/calosum/verify_v3.py` | ❌ não existe |
| `debug_numpy` | `src/calosum/debug_numpy.py` | ❌ não existe |

Também existe uma regra `domain.tool_registry` sem arquivo correspondente:

| Regra morta | Arquivo esperado | Status |
|---|---|---|
| `domain.tool_registry` | `src/calosum/domain/tool_registry.py` | ❌ não existe |

**Risco:** As ghost rules não causam falha hoje (o harness só falha em `missing_module_rule` para módulos
existentes sem regra, não o inverso), mas geram ruído e confundem agentes e devs sobre o inventário real.

---

### Problema 2 — `docs/ARCHITECTURE.md` desatualizado (22 módulos faltando na listagem)

A listagem de módulos no `ARCHITECTURE.md` está desatualizada. Módulos implementados e registrados
no harness não aparecem na documentação:

**domain/** — ausentes do ARCHITECTURE.md:
- `agent_config.py`
- `differentiable_logic.py`
- `directive_guardrails.py`
- `evolution.py`
- `execution_utils.py`
- `idle_foraging.py`
- `introspection.py`
- `introspection_capabilities.py`
- `self_model.py`
- `workspace.py`

**adapters/** — ausentes do ARCHITECTURE.md:
- `bridge_cross_attention.py`
- `adapters/infrastructure/contract_wrappers.py`
- `gea_experience_distributed.py`
- `gea_experience_store.py`
- `gea_reflection_experience.py`
- `latent_exchange.py`
- `left_hemisphere_rlm.py`
- `multimodal_perception.py`
- `right_hemisphere_jepars.py`
- `right_hemisphere_vjepa21.py`
- `right_hemisphere_vljepa.py`
- `telemetry_otlp.py`

**bootstrap/** — ausente do ARCHITECTURE.md:
- `backend_resolvers.py`

---

### Problema 3 — Módulos críticos no limite de 400 linhas (zona de risco)

Monitoramento ativo necessário:

| Módulo | Linhas | Status |
|---|---|---|
| `domain/orchestrator.py` | **400** | 🔴 no limite exato |
| `adapters/llm_qwen.py` | 399 | 🟠 1 linha da violação |
| `domain/agent_execution.py` | 396 | 🟠 4 linhas da violação |
| `domain/metacognition.py` | 395 | 🟠 5 linhas da violação |
| `domain/evolution.py` | 394 | 🟠 6 linhas da violação |
| `adapters/memory_qdrant.py` | 393 | 🟠 7 linhas da violação |
| `shared/types.py` | 391 | 🟡 9 linhas da violação |
| `adapters/infrastructure/contract_wrappers.py` | 387 | 🟡 |
| `domain/telemetry.py` | 379 | 🟡 |
| `bootstrap/settings.py` | 354 | 🟢 confortável |

Ação imediata: documentar estratégia de extração para os 🔴/🟠 antes da próxima sprint de features.

---

### Problema 4 — Artefatos de diagnóstico vivendo fora de `examples/`

Os seguintes scripts estão em `examples/` mas foram citados como módulos raiz no harness (`MODULE_RULES`):
- `examples/final_prod_val.py` — script de validação de produção API
- `examples/right_hemisphere_benchmark.py` — benchmark de hemisferio direito
- `examples/cognitive_cycle.py` — demo de ciclo cognitivo
- `examples/api_client.py` — client de diagnóstico
- `examples/group_reflection.py` — demo de reflexão em grupo

O harness tentava registrar `final_prod_val`, `verify_v3`, `debug_numpy` como módulos do *pacote*,
quando na verdade são scripts externos. A limpeza já foi parcial (os scripts foram movidos para
`examples/`), mas as regras mortas permanecem no harness.

---

### Problema 5 — `docs/ARCHITECTURE.md` não descreve as camadas de `bootstrap/routers/` e `adapters/tools/`

As sub-camadas de routers e tools existem com módulos registrados no harness, mas a arquitetura
não as menciona como sub-pacotes semanticos. Isso obscurece a estrutura real para novos devs/agentes.

---

### Problema 6 — `docs/design-docs/architectural-evolution-sota.md` desatualizado

O documento referencia um plano "Active: `docs/exec-plans/active/2026-03-30-dspy-self-learning.md`"
que não existe como arquivo ativo — o plano geral foi concluído e movido para `completed/`.
Statuses como "Proposed Future Plan" para bidirectional bridge e idle foraging estão desatualizados:
ambos já foram implementados (`domain/idle_foraging.py`, `adapters/bridge_cross_attention.py`).

---

### Problema 7 — `docs/production-roadmap.md` muito esparso

O roadmap atual tem apenas 16 linhas e não reflete os sprints concluídos nem a situação real
de produção. Precisa de atualização para registrar o que foi entregue e o que é próximo passo real.

---

### Problema 8 — `docs/exec-plans/completed/` incompleto

O plano `2026-03-30-dual-hemisphere-100-implementation-plan.md` foi movido para `completed/` mas
o campo `## Progress` ainda lista os sprints com checkboxes vazios. Indica que o arquivo foi
movido prematuramente sem o summary final.

---

### Problema 9 — Aviso de UserWarning não suprimido em `bridge_cross_attention.py`

Durante os testes, ocorre:
```
UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first.
```
Na linha `entropy = float(-torch.sum(attn * torch.log(attn + 1e-9)))`. Fácil correção.

---

### Problema 10 — `pip_install_log.txt` e `test_full_log.txt` e `test_output.txt` na raiz

Arquivos de log de diagnóstico vivendo na raiz do repositório. Devem ser removidos ou adicionados
ao `.gitignore`. Não fazem parte do código nem da documentação versionada.

---

## In Scope

1. Remover ghost rules (`final_prod_val`, `verify_v3`, `debug_numpy`, `domain.tool_registry`) do `MODULE_RULES`.
2. Atualizar `docs/ARCHITECTURE.md` com todos os módulos reais por camada.
3. Documentar sub-pacotes `adapters/tools/` e `bootstrap/routers/` na arquitetura.
4. Atualizar `docs/design-docs/architectural-evolution-sota.md` com statuses reais.
5. Enriquecer `docs/production-roadmap.md` com entregáveis concluídos e próximos passos.
6. Completar o summary final do plano `2026-03-30-dual-hemisphere-100-implementation-plan.md`.
7. Corrigir warning de `tensor.detach()` em `bridge_cross_attention.py`.
8. Adicionar `*.txt` de log ao `.gitignore` ou remover da raiz.
9. Documentar estratégia de extração para módulos 🔴/🟠 no `tech-debt-tracker.md`.
10. Atualizar `docs/QUALITY_SCORE.md` com data e scores atuais.

## Out of Scope

- Refatoração de módulos (extração de código denso) — isso é sprint separado.
- CI remota — roadmap Q2.
- Mudanças de comportamento em runtime.

## Validation

```bash
# Gate 1: Harness continua verde após cada mudança
PYTHONPATH=src .venv/bin/python -m calosum.harness_checks

# Gate 2: Testes continuam passando
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t .

# Gate 3: Sem novos warnings em bridge_cross_attention
PYTHONPATH=src .venv/bin/python -m unittest tests.test_bridge_cross_attention -v
```

## Progress

- [x] Diagnóstico executado e documentado.
- [x] Plano criado em `docs/exec-plans/active/`.
- [x] Ghost rules removidas do `MODULE_RULES` em `harness_checks.py` (`final_prod_val`, `verify_v3`, `debug_numpy`, `domain.tool_registry`).
- [x] `docs/ARCHITECTURE.md` atualizado com lista completa de módulos (23 domain, 29 adapters, 10 bootstrap, 6 shared).
- [x] Sub-pacotes `adapters/tools/` e `bootstrap/routers/` documentados na camada de arquitetura.
- [x] `docs/design-docs/architectural-evolution-sota.md` — statuses de DSPy, bridge bidirecional e idle foraging corrigidos.
- [x] `docs/production-roadmap.md` enriquecido com Q1 concluído e Q2/Q3 reais.
- [x] Summary final adicionado ao plano completed `2026-03-30-dual-hemisphere-100-implementation-plan.md`.
- [x] Warning de `tensor.detach()` corrigido em `bridge_cross_attention.py`.
- [x] Logs de diagnóstico adicionados ao `.gitignore` (`pip_install_log.txt`, `test_full_log.txt`, `test_output.txt`).
- [x] `tech-debt-tracker.md` atualizado com módulos em zona de risco e estratégias de extração.
- [x] `docs/QUALITY_SCORE.md` atualizado (data, scores atualizados, novas integrações listadas).
- [x] Harness final verde.
- [x] Testes `test_bridge_cross_attention` passando sem warnings.
- [ ] Plano movido para `docs/exec-plans/completed/` com summary.


## Decision Log

- 2026-03-31: Ghost rules (`final_prod_val`, `verify_v3`, `debug_numpy`) foram identificadas como
  scripts que existiam na raiz do pacote durante debugging e foram movidos para `examples/` em sprint
  anterior, mas suas entradas em `MODULE_RULES` não foram limpas. Removidas.

- 2026-03-31: A estratégia de sanitização escolhida é não-destrutiva: nenhuma feature é removida,
  apenas documentação e harness são corrigidos para refletirem a realidade.

- 2026-03-31: Os módulos em zona de risco (orchestrator, llm_qwen, agent_execution) foram documentados
  como debt crítico com estratégias de extração específicas, mas a extração fica para sprint dedicada
  para não misturar sanitização com refatoração.

## Resumo Final (2026-03-31)

Sanitização concluída. 10 problemas identificados e todos resolvidos:

1. 4 ghost rules removidas de `MODULE_RULES`.
2. `ARCHITECTURE.md` atualizado com 23 módulos domain, 29 adapters, sub-pacotes tools/routers documentados.
3. `architectural-evolution-sota.md` corrigido com statuses de implementação reais.
4. `production-roadmap.md` enriquecido com histórico Q1 completo.
5. Plano completed `2026-03-30` recebeu checkboxes marcados e resumo final.
6. UserWarning de `tensor.detach()` corrigido em `bridge_cross_attention.py`.
7. Logs de diagnóstico na raiz cobertos por `.gitignore`.
8. `tech-debt-tracker.md` com tabelas de risco, estratégias e itens resolvidos.
9. `QUALITY_SCORE.md` atualizado para 2026-03-31 com novas integrações e scores.
10. Harness verde. Testes sem warnings.
