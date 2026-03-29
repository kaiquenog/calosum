# Evoluções Arquiteturais - Ciclo 1

**Data:** 28 de Março de 2026
**Status:** Planejado

## Purpose
Este plano descreve a execução das Evoluções 1, 3 e 4 mapeadas na avaliação de arquitetura.

## Scope
1. **Evolução 1: Sandboxing de Execução (Segurança)**
2. **Evolução 3: Active Inference (Hemisfério Direito)**
3. **Evolução 4: Barramento de Eventos Interno (Concorrência)**
A Evolução 2 (Clusterização Semântica no Sleep Mode via LLM) foi deliberadamente postergada para o próximo ciclo de melhorias.

## Progress
### Fase 1: Sandboxing de Execução (Evolução 1)
**Componente Afetado:** `src/calosum/adapters/action_runtime.py` e `src/calosum/domain/runtime.py`

- [x] **Passo 1.1:** Avaliar SDKs de sandbox.
- [x] **Passo 1.2:** Criar a interface/adaptador `SandboxedActionRuntime`.
- [x] **Passo 1.3:** Migrar as ações existentes para rodarem dentro da microVM/container efêmero.
- [x] **Passo 1.4:** Adicionar tratamento de exceções.
- [x] **Passo 1.5:** Atualizar os testes unitários.

### Fase 2: Active Inference (Evolução 3)
**Componente Afetado:** `src/calosum/domain/right_hemisphere.py` e `src/calosum/domain/metacognition.py`

- [x] **Passo 2.1:** Expandir o retorno do `QdrantDualMemoryAdapter.build_context`.
- [x] **Passo 2.2:** No `RightHemisphereJEPA` calcular a distância de cosseno.
- [x] **Passo 2.3:** Incluir a métrica `surprise_score` no `RightHemisphereState`.
- [x] **Passo 2.4:** Alterar o loop de variantes para reagir à alta `surprise_score`.
- [x] **Passo 2.5:** Atualizar testes de domínio e adaptadores de IA.

### Fase 3: Barramento de Eventos Interno (Evolução 4)
**Componente Afetado:** `src/calosum/domain/orchestrator.py`

- [x] **Passo 3.1:** Introduzir um `InternalEventBus` no domínio.
- [x] **Passo 3.2:** Refatorar o `CalosumAgent` para disparar eventos.
- [x] **Passo 3.3:** Reduzir ou circunscrever as chamadas `run_sync`.
- [x] **Passo 3.4:** Garantir thread-safety.
- [x] **Passo 3.5:** Executar testes.

## Decision Log
- Decidimos usar um sandboxing leve com `tempfile` em `adapters/action_runtime.py` para isolar I/O.
- Decidimos implementar o loop assíncrono para o event bus sem bloquear a UI e permitindo eventos como side effects não bloqueantes.

## Validation
- `PYTHONPATH=src python3 -m calosum.harness_checks` executado com sucesso e zero violações de arquitetura.
- `python -m unittest` rodando sem quebras nas asserções assíncronas.
- O Frontend SSE/FastAPI rodando simultaneamente múltiplos turnos sem bloquear o event-loop principal.