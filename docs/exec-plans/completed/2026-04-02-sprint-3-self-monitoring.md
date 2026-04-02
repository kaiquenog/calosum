# Sprint 3 Self Monitoring and Telemetry Query

## Purpose
Implementar automonitoramento operacional no runtime do agente para fechar o ciclo entre telemetria, decisão e ajuste de estratégia por sessão.

## Scope
- Registrar novas tools no ToolRegistry: `query_session_stats`, `explain_last_decision`, `read_architecture`, `propose_config_change`.
- Injetar Session Briefing no prompt do hemisfério esquerdo em todos os turnos.
- Persistir Cognitive Diary em `.calosum-runtime/cognitive_diary.jsonl`.
- Expor endpoint `POST /v1/telemetry/query` para consulta em linguagem natural.
- Adicionar cobertura de testes para os novos contratos.

## Validation
- `PYTHONPATH=src ./.venv/bin/python3 -m unittest tests.adapters.tools.test_tool_registry tests.bootstrap.test_api`
- Verificar `POST /v1/telemetry/query` com pergunta sobre falhas por tipo de tool.
- Verificar presença de Session Briefing no prompt montado para o LLM.
- Confirmar escrita de entradas no arquivo `.calosum-runtime/cognitive_diary.jsonl`.

## Progress
- [x] Plano ativo criado
- [x] Novas tools adicionadas ao runtime e introspection adapter
- [x] Session Briefing integrado ao prompt por turno
- [x] Endpoint `/v1/telemetry/query` implementado
- [x] Cognitive Diary persistido em arquivo JSONL
- [ ] Testes atualizados e executados

## Decision Log
- 2026-04-02: Session Briefing calculado no orquestrador para manter estratégia adaptativa independente do provider do LLM.
- 2026-04-02: `propose_config_change` usa `EvolutionManager.queue_directive` para garantir persistência pending no archive existente.
- 2026-04-02: `read_architecture` implementado com AST read-only sobre `src/calosum`.
