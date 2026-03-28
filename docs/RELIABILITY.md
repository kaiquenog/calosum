# Reliability

## Objetivos

- Todo turno deve produzir telemetria e latencia observavel.
- Rejeicoes do runtime devem acionar reparo automatico limitado.
- O repositorio deve falhar cedo quando arquitetura ou documentacao derivarem.

## Controles Atuais

- `CalosumAgentConfig.max_runtime_retries`
- `StrictLambdaRuntime` com rejeicao explicita
- `CognitiveTelemetryBus` com `trace_id`, `span_id` e metricas
- Harness checks para docs, planos e fronteiras de importacao

## SLOs Iniciais

- `process_turn`: telemetria obrigatoria por turno
- `runtime_retry_count`: observavel por decisao
- `sleep_mode`: nao pode perder regras ou triplas ja consolidadas

## Operacao

- Rodar harness checks antes de mudancas estruturais
- Rodar testes antes de merge
- Registrar debt estrutural no tracker
