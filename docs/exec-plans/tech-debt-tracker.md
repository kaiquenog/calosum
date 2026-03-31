# Tech Debt Tracker

Catalogo ativo de debitos tecnicos que surgem durante execucao de planos.
Atualizado em: 2026-03-31

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
| `adapters/contract_wrappers.py` | 387 | 2026-03-31 |
| `domain/telemetry.py` | 379 | 2026-03-31 |

### 🟡 Pendentes de sprint dedicada

1. **2026-03-30** — CI remota executando harness checks e testes em cada PR. Impacto: médio. Bloqueia detecção automática de regressões arquiteturais.
2. **2026-03-30** — Exportador OTLP direto para collector externo. Impacto: baixo em dev, médio em prod.
3. **2026-03-30** — Validação de docstrings de `__init__.py` no harness. Impacto: baixo.
4. **2026-03-30** — Benchmark do hemisferio direito com checkpoint HF real pré-carregado. Impacto: médio.
5. **2026-03-31** — `shared/ports.py` depende de `domain.metacognition` (importação para type hint). Avaliar se pode ser movida para `shared/types.py` para quebrar essa dependência reversa. Impacto: baixo.
6. **2026-03-31** — Warning `MPLCONFIGDIR` durante testes (matplotlib tempdir). Adicionar `MPLCONFIGDIR` ao entrypoint de testes ou `.env`.

## Regra de Atualização

- Todo debt novo identificado em sprint deve entrar aqui com data, impacto e estratégia.
- Itens concluídos devem ser movidos para seção `## Resolvidos` com data de fechamento.

## Resolvidos

- **2026-03-31** — Ghost rules `final_prod_val`, `verify_v3`, `debug_numpy` e `domain.tool_registry` removidas do `MODULE_RULES`.
- **2026-03-31** — Warning `tensor.detach()` em `bridge_cross_attention.py` corrigido.
- **2026-03-30** — Contratos de backend padronizados via `adapters/contract_wrappers.py`.

