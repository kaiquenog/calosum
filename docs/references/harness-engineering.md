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

Conforme a base de código cresce, a fragmentação de dezenas de módulos na mesma pasta aumenta a "entropia" (confundindo desenvolvedores humanos e agentes IA focados na manutenabilidade). Para isso, implementou-se:

1. **Subdiretórios Estritos**: Todo arquivo Python agora deve residir no subdiretório semântico correto (`shared`, `domain`, `adapters`, `bootstrap`).
2. **Invariantes por Pacote**: Em vez de arquivos silenciosos, o `__init__.py` de todo pacote contém uma declaração docstring oficial explicando o que o pacote **não pode fazer**, balizando o escopo de imediato.
3. **Checagem Recursiva**: O `harness_checks.py` foi atualizado para varrer tudo recursivamente e validar injeção de dependência hierárquica usando FQDN (*Fully Qualified Domain Names*) para os módulos.
