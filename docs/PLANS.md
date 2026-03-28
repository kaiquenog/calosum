# Plans

Planos sao artefatos de primeira classe do repositorio.

## Quando Abrir Um Plano

- Mudanca que toca mais de um subsistema
- Nova integracao externa
- Refatoracao estrutural
- Mudanca de confiabilidade, memoria, telemetria ou runtime

## Template Minimo

Todo plano em `docs/exec-plans/active/` ou `docs/exec-plans/completed/` deve conter:

- `# Title`
- `## Purpose`
- `## Scope`
- `## Validation`
- `## Progress`
- `## Decision Log`

## Regras

- Planos ativos descrevem o estado atual do trabalho.
- Ao finalizar, mova o plano para `completed/` com um resumo final.
- Debt identificado durante a execucao deve ir para `exec-plans/tech-debt-tracker.md`.
