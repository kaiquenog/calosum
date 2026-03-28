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

- O harness nao exige hoje que todo arquivo Python esteja dentro dos quatro subpacotes semanticos.
- O harness nao valida ainda a presenca ou o conteudo das docstrings de `__init__.py`.
- O harness verifica fronteiras de importacao a partir de `MODULE_RULES`; quando a estrutura muda, esse mapa e a documentacao precisam evoluir juntos.
