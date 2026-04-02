# Sprint 6.1-6.3 - Qualidade de Producao em Paralelo
## Purpose
Estabelecer gates de qualidade no CI/CD, consolidar modo local JEPA no Docker e manter documentacao viva em sincronia com a arquitetura real para evitar acumulacao de divida tecnica durante desenvolvimento de features.

## Scope
- Evoluir `.github/workflows/ci.yml` para estagios com gates de qualidade:
  - lint + types (`mypy --strict`, `ruff`, `harness_checks`)
  - unit tests com gate de cobertura para modulos novos/alterados
  - integration com mocks de LLM e gate de latencia p95 <= 5s no perfil `ephemeral`
  - benchmark gate automatico com regressao maxima de 5% em `tool_success_rate`
- Adicionar scripts de suporte para gates de integracao/benchmark/cobertura em `scripts/`.
- Atualizar `deploy/docker-compose.yml` para modo local com JEPA via volume read-only e variaveis de ambiente.
- Atualizar `README.md` e `docs/ARCHITECTURE.md` para refletir operacao real dos gates e artefatos de benchmark de CI.
- Criar `ARCHITECTURE.md` proprio para os componentes renomeados no Sprint 0 e registrar aliases.
- Garantir producao automatica de resultados de benchmark em `docs/benchmarks/` durante CI (artefatos de run).

## Validation
- `PYTHONPATH=src ./.venv/bin/python3 -m calosum.harness_checks`
- `PYTHONPATH=src ./.venv/bin/python3 -m unittest tests/bootstrap/test_settings_dependency_mode.py tests/integration/test_pipeline.py tests/integration/test_pipeline_dual_hemisphere_e2e.py`
- `PYTHONPATH=src ./.venv/bin/python3 scripts/ci_integration_benchmark.py --output-json docs/benchmarks/ci/local_probe.json --output-md docs/benchmarks/ci/local_probe.md --latency-p95-threshold-ms 5000`
- `PYTHONPATH=src ./.venv/bin/python3 scripts/ci_benchmark_gate.py --baseline docs/benchmarks/ci/baseline.json --candidate docs/benchmarks/ci/local_probe.json --metric tool_success_rate --max-regression-percent 5.0`

## Progress
- [x] Plano criado e alteracoes implementadas em CI/Docker/docs/scripts.
- [x] Gates de qualidade executando em pipeline CI.
- [x] Documentacao viva atualizada com arquitetura real e benchmark de CI.

## Decision Log
- 2026-04-02: Gates de lint/types e cobertura passam a focar modulos alterados para viabilizar adocao incremental sem bloquear backlog por divida preexistente.
- 2026-04-02: Benchmark de CI usa pipeline com LLM mockado para garantir reproducibilidade sem dependencias externas.

## Summary
- CI/CD agora possui gates explicitos para lint/types, cobertura de modulos alterados, latencia p95 e regressao de `tool_success_rate`.
- Docker local foi alinhado ao modo JEPA com volume read-only de modelo e env vars dedicadas.
- Documentacao principal e catalogos de benchmark/componentes foram atualizados para refletir o estado real.
