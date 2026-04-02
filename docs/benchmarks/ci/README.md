# CI Benchmarks

Esta pasta recebe artefatos automaticos do pipeline de CI em cada execucao:

- `integration_latest.json` / `integration_latest.md`: latencia e `tool_success_rate` do pipeline de integracao em modo ephemeral com LLM mockado.
- `baseline.json`: baseline versionado para o gate de regressao de benchmark.

No CI, esses arquivos sao gerados/atualizados no workspace do run e publicados como artefatos.
