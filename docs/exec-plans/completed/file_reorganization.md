# File Reorganization and Harness Docs

## Purpose
Adaptar a base de código do Calosum para padrões de " baixa entropia " do Harness Engineering. Isso envolverá estruturar os `.py` soltos em sub-pastas por domínio e adicionar docstrings de invariants (`__init__.py`) em cada módulo.

## Scope
Pastas: `src/calosum/`
Módulos Coretos: `shared`, `domain`, `adapters`, `bootstrap`.
Arquivos Docs: `docs/ARCHITECTURE.md`, `docs/references/harness-engineering.md`.

## Validation
O script `harness_checks.py` será substancialmente alterado para aplicar um `rglob` recursivo. A validação será um "Exit Code 0" indicando pureza das dependências importadas e a passagem da suite de Unit Tests inteira.

## Progress
Planning phase. Planejando a movimentação de arquivos e re-escrita dos imports.

## Decision Log
Decidimos usar a seguinte topologia:
- `shared`: Utilitários puros, tipos, protocolos (Ports).
- `domain`: Regras de negócio, hemisférios, bridge, orchestrator, execution engine.
- `adapters`: Integrações baseadas em abstração.
- `bootstrap`: Arquivos de entrada, CLI, Factory e Setup.
