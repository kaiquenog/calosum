# CI Benchmarks

Esta pasta recebe artefatos automaticos do pipeline de CI em cada execucao:

- `integration_latest.json` / `integration_latest.md`: latencia e `tool_success_rate` do pipeline de integracao em modo ephemeral com LLM mockado.
- `baseline.json`: baseline versionado para o gate de regressao de benchmark.
- `2026-04-03-reflection-branching-smoke.*`: smoke explicitando `candidate_count >= 2`, latencia e contrato `GroupTurnResult.selected_result`.
- `2026-04-03-docker-profile-ready.*`: smoke versionado do payload de readiness do perfil `docker`.

No CI, esses arquivos sao gerados/atualizados no workspace do run e publicados como artefatos.
