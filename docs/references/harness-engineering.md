# Harness Engineering References

## Fontes

- OpenAI, "Harness engineering: leveraging Codex in an agent-first world" (11 de fevereiro de 2026)  
  https://openai.com/index/harness-engineering/
- OpenAI Cookbook, tópico "Agents" com referência a `PLANS.md` e eval-driven system design  
  https://cookbook.openai.com/topic/agents

## Princípios extraídos

- Use um `AGENTS.md` curto como mapa, não como enciclopédia.
- Trate `docs/` como sistema de registro versionado.
- Transforme planos em artefatos permanentes, não contexto oral.
- Enforce arquitetura e *taste* com checks mecânicos, não revisão ad hoc.
- Capture *debt* e lixo continuamente, em vez de limpezas episódicas.
- Favoreça estruturas simples, previsíveis e legíveis para agentes.

## Aplicação em Calosum

- `AGENTS.md` curto apontando para `docs/`
- Índice e *scorecards* em `docs/`
- Harness checks mecânicos em `src/calosum/harness_checks.py`
- Planos versionados em `docs/exec-plans/`
- *Tracker* de débito em `docs/exec-plans/tech-debt-tracker.md`

## Controle de entropia modular

Conforme a base de código cresce, a fragmentação de dezenas de módulos na mesma pasta aumenta a “entropia” (confundindo desenvolvedores humanos e agentes focados em manutenibilidade). No Calosum:

1. **Pacotes semânticos como padrão:** O código de produto fica em `shared/`, `domain/`, `adapters/` e `bootstrap/`. Utilitários de governança podem ficar na raiz do pacote quando simplifica a execução — é o caso de `src/calosum/harness_checks.py`.
2. **Docstrings de fronteira por pacote semântico:** Os subpacotes carregam `__init__.py` com papel e invariante de design. O harness exige **docstring de módulo** nesses `__init__.py` para os caminhos listados em `SEMANTIC_PACKAGES` (ver código-fonte).
3. **Regras FQDN:** O harness valida fronteiras de importação com nomes de módulo qualificados (`domain.agent.orchestrator`, `bootstrap.entry.cli`, …) no mapa `MODULE_RULES`.

## Limites atuais do harness

- Não valida que **todo** arquivo Python esteja apenas nos quatro subpacotes semânticos; `harness_checks.py` fica na raiz do pacote por simplicidade e está **isento** do limite de 400 linhas (assim como `__init__.py` não entram na contagem de tamanho — apenas `.py` “normais”).
- Quando a estrutura muda (novos módulos, renomeações), `MODULE_RULES` **e** esta documentação devem evoluir juntos. Módulos não registrados geram `missing_module_rule` e quebram o *gate*.
- O harness não substitui testes, Ruff ou mypy: no CI eles rodam em estágios distintos (ver abaixo).

## Como executar

```bash
PYTHONPATH=src python3 -m calosum.harness_checks
# ou, após pip install -e .:
calosum-harness
```

Saída esperada em sucesso: `Harness checks passed.` (código de saída `0`).

Implementação: `run_harness_checks()` em `src/calosum/harness_checks.py` — ordem das verificações:

1. `_check_required_paths`
2. `_check_agents_map`
3. `_check_docs_index`
4. `_check_plan_files`
5. `_check_module_sizes`
6. `_check_import_boundaries` (`MODULE_RULES` + AST)
7. `_check_package_docstrings`
8. `_check_shared_domain_runtime_imports`
9. `_check_adapter_isolation`
10. `_check_forbidden_domain_patterns`

## Códigos de issue (`HarnessIssue.code`)

| Código | Descrição |
|--------|-----------|
| `missing_required_path` | Falta um arquivo listado em `REQUIRED_PATHS` (ex.: `AGENTS.md`, docs obrigatórios). |
| `missing_agents_map` | `AGENTS.md` ausente na raiz do repositório. |
| `agents_too_long` | `AGENTS.md` excedeu 120 linhas. |
| `agents_missing_link` | `AGENTS.md` não contém um dos links obrigatórios (`REQUIRED_DOC_LINKS`). |
| `missing_docs_index` | `docs/index.md` ausente. |
| `docs_index_missing_ref` | `docs/index.md` não referencia um doc obrigatório (lista no código). |
| `missing_plan_directory` | Falta `docs/exec-plans/active` ou `completed`. |
| `plan_missing_heading` | Plano `.md` sem um dos headings exigidos (`PLAN_REQUIRED_HEADINGS`). |
| `module_too_large` | Módulo `.py` com mais de 400 linhas (`harness_checks.py` isento). |
| `missing_module_rule` | Módulo não registrado em `MODULE_RULES`. |
| `forbidden_internal_import` | Import interno `calosum.*` fora do conjunto permitido para aquele módulo. |
| `missing_package_docstring` | `__init__.py` de um pacote em `SEMANTIC_PACKAGES` sem docstring de módulo. |
| `shared_domain_runtime_import` | Código em `shared/` importa `domain.*` fora de bloco `TYPE_CHECKING`. |
| `forbidden_adapter_import` | Adapter importa `os` ou `subprocess` diretamente (exceto lista de isenção no código). |
| `forbidden_domain_pattern` | Arquivo em `domain/` contém padrões proibidos de ML/treino (`torch.`, `nn.Module`, `train(`). |

## CI (GitHub Actions)

O workflow [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) é uma **cadeia de estágios**, não um único job de harness:

| Job | Conteúdo principal |
|-----|---------------------|
| `lint_types` | `harness_checks`, Ruff (arquivos `.py` alterados), mypy `--strict` (módulos `src/calosum` alterados). |
| `unit_tests` | `unittest discover` + *coverage* + *gate* 80% em módulos novos/alterados. |
| `integration` | Benchmark de integração com limite `p95` de latência (perfil `ephemeral`). |
| `benchmark_gate` | Comparação contra *baseline* versionado (`tool_success_rate`, regressão máx. 5%). |

Artefatos de benchmark são gravados em `docs/benchmarks/ci/` e publicados como *artifacts* do CI.

## Benchmarks manuais (hemisfério direito)

O script `scripts/benchmark_right_hemisphere.py` mede latência e qualidade do hemisfério direito Hugging Face com um checkpoint real.

```bash
# Rodar benchmark (CPU-only, ~2–5 min na primeira execução pelo download do modelo)
PYTHONPATH=src .venv/bin/python3 scripts/benchmark_right_hemisphere.py
```

**Saída:** relatório datado em `docs/reports/benchmark_right_hemi_YYYY-MM-DD.md` com latência (média, mediana, P95, min, max) em ms, `surprise_score` médio e desvio padrão, tempo de carregamento do modelo.

**Baseline:** referência para comparação futura entre backends (`huggingface`, `vjepa21`, `jepars`).
