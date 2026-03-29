# Docker Validation Fixes

## Purpose

Corrigir os desvios encontrados na validacao Docker apos rebuild do stack:

- `/v1/system/introspect` ignora `session_id` e responde sem contexto real da sessao.
- `pending_approval_backlog` mistura fila de diretivas com aprovacoes de runtime.
- O stack expõe `otel-collector` e `jaeger`, mas a telemetria cognitiva nao chega como traces no collector.

## Scope

- Ajustar a API e a tool `introspect_self` para consumir a sessao solicitada.
- Separar backlog de aprovacao de runtime da fila de diretivas pendentes no diagnostico.
- Adicionar exportacao OTLP HTTP util ao barramento de telemetria sem perder o sink JSONL local.
- Cobrir os cenarios com testes unitarios/integracao e revalidar no Docker.

## Validation

- `PYTHONPATH=src python3 -m calosum.harness_checks`
- `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t .`
- `docker compose -f deploy/docker-compose.yml restart orchestrator`
- Chamar `/v1/system/introspect`, `/v1/system/awareness`, `/v1/chat/completions`, `/v1/system/state` e consultar `jaeger`

## Progress

- [x] Ajustar contexto de sessao para introspeccao via API/tool
- [x] Corrigir semantica de backlog no diagnostico de awareness
- [x] Exportar traces OTLP para o collector mantendo persistencia JSONL
- [x] Revalidar com testes e smoke test Docker

## Decision Log

- 2026-03-29: O fix deve preservar o dashboard local em JSONL e adicionar tracing distribuido como espelho, nao como substituicao.
- 2026-03-29: `pending_approval_backlog` passa a representar apenas aprovacoes reais de runtime; a fila de evolucao fica em `pending_directive_count`.

## Completion Summary

- `system_introspect` agora encaminha `session_id` e a tool `introspect_self` usa a sessao resolvida para workspace, awareness e respostas grounded de backend/memoria/telemetria.
- `workspace_for_session()` e `latest_awareness_for_session()` deixaram de vazar o ultimo estado quando uma sessao inexistente e solicitada.
- O diagnostico separa backlog de aprovacao e fila de diretivas, e a awareness persistida/exportada carrega ambos os sinais.
- O barramento de telemetria agora usa sink composto: JSONL consultavel para dashboard e OTLP HTTP para `otel-collector`/`jaeger`.
- Validacao final: `jaeger` passou a expor o servico `calosum`, `/v1/system/state?session_id=missing-session` devolveu `404`, e a introspeccao da sessao `docker-fix-validation` refletiu o runtime correto.
