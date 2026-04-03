# Tech Debt Tracker

Catalogo ativo de debitos tecnicos que surgem durante execucao de planos.
Atualizado em: 2026-04-03

## Itens Abertos

### 🔴 Crítico — Módulos no limite exato de 400 linhas

| Módulo | Linhas | Data | Estratégia de extração |
|---|---|---|---|
| `domain/orchestrator.py` | 400 | 2026-03-31 | Extrair fluxo de group turn para `domain/orchestrator_group.py` |
| `adapters/llm_qwen.py` | 399 | 2026-03-31 | Extrair handlers de payload para `adapters/llm_qwen_payloads.py` |
| `domain/agent_execution.py` | 396 | 2026-03-31 | Extrair lógica de repair para `domain/execution_repair.py` |
| `domain/metacognition.py` | 395 | 2026-03-31 | Extrair scorer/reward engine para `domain/metacognition_scorer.py` |
| `domain/evolution.py` | 394 | 2026-03-31 | Extrair funções de mutação para `domain/evolution_ops.py` |

**Regra:** Qualquer nova feature nesses módulos EXIGE extração prévia na mesma PR.

### 🟠 Atenção — Módulos em zona de risco (>380 linhas)

| Módulo | Linhas | Data |
|---|---|---|
| `adapters/memory_qdrant.py` | 393 | 2026-03-31 |
| `shared/types.py` | 391 | 2026-03-31 |
| `adapters/infrastructure/contract_wrappers.py` | 387 | 2026-03-31 |
| `domain/telemetry.py` | 379 | 2026-03-31 |

### 🟡 Pendentes de sprint dedicada

> Nenhum item pendente. Todos os débitos técnicos levantados até 2026-03-31 foram resolvidos.

## Regra de Atualização

- Todo debt novo identificado em sprint deve entrar aqui com data, impacto e estratégia.
- Itens concluídos devem ser movidos para seção `## Resolvidos` com data de fechamento.

## Resolvidos

- **2026-03-31** — CI remota (`.github/workflows/ci.yml`): pipeline em estágios (`lint_types` → `unit_tests` → `integration` → `benchmark_gate`) em push/PR para `main`.
- **2026-03-31** — OTLP hardening: documentação ponta-a-ponta adicionada ao `INFRASTRUCTURE.md`; fallback gracioso confirmado via teste.
- **2026-03-31** — Docstrings no harness: check `missing_package_docstring` adicionado; `bootstrap/routers/__init__.py` corrigido.
- **2026-03-31** — Benchmark hemisfério direito: `scripts/benchmark_right_hemisphere.py` criado; documentado em `harness-engineering.md`.
- **2026-03-31** — Dependência reversa `shared/ports.py → domain.metacognition`: confirmada dentro de `TYPE_CHECKING` (sem impacto runtime); check `shared_domain_runtime_import` adicionado ao harness para enforcement futuro.
- **2026-03-31** — Warning `MPLCONFIGDIR`: corrigido em `tests/__init__.py` (compatível com `unittest discover` e pytest).
- **2026-03-31** — Ghost rules `final_prod_val`, `verify_v3`, `debug_numpy` e `domain.tool_registry` removidas do `MODULE_RULES`.
- **2026-03-31** — Warning `tensor.detach()` em `bridge_cross_attention.py` corrigido.
- **2026-03-30** — Contratos de backend padronizados via `adapters/infrastructure/contract_wrappers.py`.
