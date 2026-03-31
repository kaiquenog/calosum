# Harness Engineering References

## Fontes

- OpenAI, "Harness engineering: leveraging Codex in an agent-first world" (11 de fevereiro de 2026)
  https://openai.com/index/harness-engineering/
- OpenAI Cookbook, topico "Agents" com referencia a `PLANS.md` e eval-driven system design
  https://cookbook.openai.com/topic/agents

## Principios Extraidos

- Use um `AGENTS.md` curto como mapa, nao como enciclopedia.
- Trate `docs/` como sistema de registro versionado.
- Transforme planos em artefatos permanentes, nao contexto oral.
- Enforce arquitetura e taste com checks mecanicos, nao revisao ad hoc.
- Capture debt e lixo continuamente, em vez de limpancas episodicas.
- Favoreca estruturas simples, previsiveis e legiveis para agentes.

## Aplicacao em Calosum

- `AGENTS.md` curto apontando para docs
- docs indexados e scorecards
- harness checks mecanicos
- planos versionados
- tracker de debt

## Controle de Entropia Modular

Conforme a base de código cresce, a fragmentação de dezenas de módulos na mesma pasta aumenta a "entropia" (confundindo desenvolvedores humanos e agentes IA focados na manutenabilidade). No estado atual do Calosum, a organizacao aplicada e:

1. **Pacotes Semanticos Como Default**: O codigo de produto fica organizado prioritariamente em `shared`, `domain`, `adapters` e `bootstrap`. Utilitarios de governanca do repositorio podem continuar no nivel raiz do pacote quando isso simplifica a execucao, como ocorre com `src/calosum/harness_checks.py`.
2. **Docstrings de Fronteira por Pacote Semantico**: Os subpacotes semanticos carregam `__init__.py` com o papel e a invariante de design do pacote. O `src/calosum/__init__.py` permanece focado na surface publica do pacote, nao em policiamento arquitetural.
3. **Checagem Recursiva com Regras FQDN**: O `harness_checks.py` varre recursivamente `src/calosum` e valida artefatos obrigatorios, links minimos de docs, formato de planos, tamanho maximo de modulos e fronteiras de importacao via nomes de modulo qualificados (*Fully Qualified Domain Names*).

## Limites Atuais do Harness

- O harness não valida que todo arquivo Python esteja dentro dos quatro subpacotes semânticos; `harness_checks.py` é mantido no nível raiz do pacote por simplicidade de execução e está **isento** do limite de 400 linhas.
- O harness verifica fronteiras de importação a partir de `MODULE_RULES`; quando a estrutura muda (novos módulos, renomeações), esse mapa **e** esta documentação precisam evoluir juntos. Módulos não registrados em `MODULE_RULES` geram violação `missing_module_rule` que quebra o build.
- Em CI remota (GitHub Actions), os dois jobs (`harness` + `tests`) rodam em paralelo em cada push/PR para `main`. Ver `.github/workflows/ci.yml`.

## Checks Atuais (harness_checks.py)

| Check | Descrição |
|---|---|
| `missing_required_path` | Artefatos obrigatórios ausentes (AGENTS.md, docs/index.md, etc.) |
| `agents_too_long` | AGENTS.md excedeu 120 linhas |
| `agents_missing_link` | AGENTS.md não referencia link obrigatório de docs |
| `docs_index_missing_ref` | docs/index.md não referencia doc obrigatório |
| `plan_missing_heading` | Plano em active/ ou completed/ sem heading obrigatório |
| `module_too_large` | Módulo .py com mais de 400 linhas (harness_checks.py isento) |
| `missing_module_rule` | Módulo não registrado em MODULE_RULES |
| `forbidden_internal_import` | Módulo importa pacote não permitido pela regra de fronteira |
| `missing_package_docstring` | `__init__.py` de pacote semântico sem docstring de módulo |
| `shared_domain_runtime_import` | Módulo em `shared/` importa de `domain.*` fora de `TYPE_CHECKING` |

## Benchmarks

O script `scripts/benchmark_right_hemisphere.py` mede latência e qualidade do hemisfério direito HuggingFace com um checkpoint real.

```bash
# Rodar benchmark (CPU-only, ~2-5 min na primeira execução pelo download do modelo)
PYTHONPATH=src .venv/bin/python3 scripts/benchmark_right_hemisphere.py
```

**Saída:** relatório datado em `docs/reports/benchmark_right_hemi_YYYY-MM-DD.md` com:
- Latência (média, mediana, P95, min, max) em ms
- `surprise_score` médio e desvio padrão
- Tempo de carregamento do modelo

**Baseline:** usar como referência para comparação futura entre backends (`huggingface`, `vjepa21`, `jepars`).
