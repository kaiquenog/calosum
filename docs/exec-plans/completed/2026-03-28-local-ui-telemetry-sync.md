# Local UI Telemetry Sync

## Purpose

Corrigir o fluxo local em que `python3 -m calosum.bootstrap.cli chat` produz turnos, mas a UI nao consegue visualizar a telemetria correspondente.

## Scope

- tornar a telemetria persistida consultavel entre processos
- alinhar defaults locais de CLI chat, API e UI para a mesma sessao/telemetria
- atualizar documentacao operacional
- adicionar testes de regressao para o fluxo

## Validation

- `PYTHONPATH=src python3 -m calosum.harness_checks`
- `PYTHONPATH=src python3 -m unittest discover -s tests -t .`
- `npm run lint`
- `npm run build`

## Progress

- iniciado em 2026-03-28
- concluido em 2026-03-28
- telemetria OTLP JSONL passou a ser reidratada do disco para consultas cross-process
- `cli chat` e API local passaram a adotar persistencia local por default quando nada explicito foi configurado
- UI passou a observar `terminal-session` por default, persistir o session id no navegador e fazer polling automatico
- testes e sanitizacao executados ao final

## Decision Log

- decidido preservar `ephemeral` apenas quando o usuario o pedir explicitamente; para chat local e API local sem configuracao, a observabilidade persistente tem prioridade
- decidido reutilizar o arquivo `.calosum-runtime/telemetry/events.jsonl` como ponte entre CLI e API, em vez de introduzir um novo backend de sincronizacao
