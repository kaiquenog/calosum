# AGENTS.md

Este arquivo e um mapa curto do repositorio. Nao e a documentacao completa.

## Comece Aqui

- Leia [docs/index.md](docs/index.md) para navegar o conhecimento versionado do projeto.
- Leia [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) antes de mudar dependencias, camadas ou pontos de integracao.
- Leia [docs/PLANS.md](docs/PLANS.md) antes de iniciar mudancas maiores que um arquivo, mudancas cross-cutting ou qualquer alteracao de arquitetura.
- Consulte [docs/QUALITY_SCORE.md](docs/QUALITY_SCORE.md) e [docs/RELIABILITY.md](docs/RELIABILITY.md) ao tocar confiabilidade, runtime, memoria, telemetria ou operacao.
- Leia [docs/INFRASTRUCTURE.md](docs/INFRASTRUCTURE.md) ao tocar bootstrap, perfis ou `docker-compose`.

## Regras de Trabalho

- Mantenha o conhecimento importante dentro do repositorio. Conversas externas nao contam como fonte de verdade.
- Para mudancas maiores, crie ou atualize um plano em `docs/exec-plans/active/` e mova para `completed/` ao terminar.
- Preserve as fronteiras descritas em `docs/ARCHITECTURE.md`. Quando uma regra recorrente surgir, promova-a para checagem mecanica.
- Execute `PYTHONPATH=src python3 -m calosum.harness_checks` antes de concluir mudancas estruturais.
- Execute `PYTHONPATH=src python3 -m unittest discover -s tests -t .` antes de fechar tarefas de codigo.

## Atalhos

- Arquitetura: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Planos: [docs/PLANS.md](docs/PLANS.md)
- Qualidade: [docs/QUALITY_SCORE.md](docs/QUALITY_SCORE.md)
- Confiabilidade: [docs/RELIABILITY.md](docs/RELIABILITY.md)
- Infraestrutura: [docs/INFRASTRUCTURE.md](docs/INFRASTRUCTURE.md)
- Roadmap de producao: [docs/production-roadmap.md](docs/production-roadmap.md)
- Referencias de harness engineering: [docs/references/harness-engineering.md](docs/references/harness-engineering.md)
