# Runtime Contract Audit Application

## Purpose

Permitir a aplicacao segura da diretiva `audit_runtime_contracts` quando falhas estruturais do runtime forem detectadas, preservando o bloqueio para mudancas topologicas reais.

## Scope

- `src/calosum/domain/orchestrator.py`
- `src/calosum/domain/directive_guardrails.py`
- `src/calosum/adapters/action_runtime.py`
- `src/calosum/shared/tools.py`
- `tests/test_awareness.py`
- `tests/test_tool_registry.py`
- `tests/test_api.py`

## Validation

- `PYTHONPATH=src .venv/bin/python -m unittest tests.test_awareness tests.test_tool_registry tests.test_runtime`
- `PYTHONPATH=src python3 -m calosum.harness_checks`
- `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t .`

## Progress

- [x] Criar plano ativo
- [x] Implementar aplicacao segura da diretiva de auditoria no orchestrator
- [x] Implementar auditoria explicita de contratos no action runtime
- [x] Adicionar/ajustar testes
- [x] Executar validacoes
- [x] Mover plano para `completed/` com resumo final

## Decision Log

- A diretiva continua com tipo `TOPOLOGY`, mas apenas o subtipo read-only `action_runtime/audit_runtime_contracts` e permitido.
- Mudancas topologicas reais seguem bloqueadas por guardrail (`rejected_guardrail_topology_locked`).
- A logica de aplicacao segura da diretiva foi extraida para `directive_guardrails` para manter modulos abaixo do limite de 500 linhas do harness.
- A auditoria explicita agora retorna snapshot de contratos de tools (parametros, permissoes, approvals), tipos suportados e recomendacoes orientadas a `validation_failed`.
