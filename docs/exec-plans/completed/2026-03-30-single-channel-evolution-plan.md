# Title
Evolucao Executavel do Calosum (Sem Multicanal)

## Purpose
Converter os insights do `report_openclaw_x_calosum.txt` em uma execucao de alta alavancagem para o estado atual do repositorio, focando apenas no que e implementavel e util no contexto presente (single channel), sem expandir para multicanal.

Objetivo tecnico: aumentar robustez de runtime, isolamento de contexto e seguranca operacional sem introduzir refatoracao estrutural desnecessaria.

## Scope
Este plano cobre quatro frentes implementaveis e diretamente conectadas ao codigo atual.

### Fase 1 — Session Lane (fila por sessao) no caminho de entrada
- Implementar serializacao de processamento por `session_id` no fluxo assíncrono do runtime HTTP/Telegram.
- Garantir ordem de processamento e evitar corrida quando duas mensagens chegam quase juntas na mesma sessao.
- Manter concorrencia entre sessoes diferentes (nao bloquear globalmente o agente).
- Cobrir com testes de concorrencia (ordem e ausencia de perda de mensagem).

Arquivos alvo:
- `src/calosum/bootstrap/api.py`
- (se necessario) novo modulo pequeno em `src/calosum/bootstrap/` para lane/queue
- `tests/test_api.py` e/ou novo teste dedicado de concorrencia

### Fase 2 — Isolamento real de memoria por sessao
- Ajustar estrategia de recuperacao de episodios para priorizar/filtrar por `session_id`.
- Aplicar regra tanto para persistencia JSONL quanto para adapter Qdrant.
- Evitar vazamento de contexto entre usuarios/sessoes no mesmo processo.
- Preservar fallback para cenarios com baixa quantidade de episodios na sessao (nao degradar demais recuperacao).

Arquivos alvo:
- `src/calosum/domain/persistent_memory.py`
- `src/calosum/adapters/memory_qdrant.py`
- `tests/test_memory.py`
- `tests/test_qdrant_adapter.py`

### Fase 3 — DmPolicy minima no Telegram (allowlist/open)
- Introduzir politica de admissao antes de enfileirar `UserTurn` do Telegram.
- Suportar ao menos:
  - `open`: aceita qualquer remetente (comportamento atual explicito)
  - `allowlist`: aceita apenas IDs configurados
- Registrar bloqueio em log quando mensagem for rejeitada por politica.
- Nao incluir pairing/aprovacao manual nesta fase (evitar dependencias de UX/operacao externa).

Arquivos alvo:
- `src/calosum/adapters/channel_telegram.py`
- `src/calosum/bootstrap/settings.py`
- `tests/test_runtime.py` ou novo `tests/test_telegram_channel.py`

### Fase 4 — Integracao automatica Sleep Mode -> NightTrainer via evento interno
- Emitir evento interno ao final de `sleep_mode/asleep_mode`.
- Consumir evento para disparar o `NightTrainer` automaticamente no ciclo de consolidacao.
- Preservar execucao manual via CLI como fallback operacional.
- Garantir idempotencia basica para nao rodar treinamento duplicado para o mesmo ciclo.

Arquivos alvo:
- `src/calosum/domain/orchestrator.py`
- `src/calosum/domain/event_bus.py` (se precisar ampliar semantica do evento)
- `src/calosum/bootstrap/factory.py` (injeção de dependencia do trainer/hook)
- `tests/test_night_trainer.py`
- `tests/test_async_retry_and_persistence.py` (ou novo teste de integracao curta)

### Fora de Escopo (explicitamente)
- Multicanal e hierarquia de key cross-channel.
- WebSocket control plane.
- Modelo completo `steer/collect/followup`.
- Framework amplo de hooks publicos para plugins externos.
- Compaction por token budget com flush silencioso guiado por contexto de LLM.

## Validation
- Validacao arquitetural obrigatoria:
  - `PYTHONPATH=src python3 -m calosum.harness_checks`
- Validacao de regressao:
  - `PYTHONPATH=src python3 -m unittest discover -s tests -t .`
- Validacoes focadas por fase:
  - Concorrencia por sessao (ordem preservada)
  - Isolamento de memoria (sem cross-session retrieval indevido)
  - Bloqueio Telegram por allowlist
  - Disparo automatico NightTrainer apos consolidacao

## Progress
- [x] Consolidar analise tecnica do report contra estado real do codigo.
- [x] Definir escopo implementavel sem multicanal.
- [x] Implementar Fase 1 (Session Lane).
- [x] Implementar Fase 2 (isolamento de memoria por sessao).
- [x] Implementar Fase 3 (DmPolicy minima no Telegram).
- [x] Implementar Fase 4 (evento Sleep Mode -> NightTrainer).
- [x] Executar harness + testes e registrar evidencias.
- [x] Mover plano para `docs/exec-plans/completed/` com resumo final.

## Decision Log
- 2026-03-30: O plano foi restrito a single channel por diretriz do solicitante; propostas de multicanal foram descartadas nesta execucao.
- 2026-03-30: Priorizacao definida por impacto operacional imediato: robustez de concorrencia, isolamento de contexto, seguranca de entrada e automacao de aprendizado continuo.
- 2026-03-30: Itens de maior incerteza de produto (WebSocket/hook framework/steering modes) ficaram fora para evitar overengineering sem demanda validada.
- 2026-03-30: Session lane foi aplicado no caminho API/Telegram com lock por sessao, preservando concorrencia entre sessoes.
- 2026-03-30: Isolamento de memoria foi implementado na consulta episodica (in-memory, JSONL e Qdrant), com fallback para pool global quando a sessao ainda nao tem historico.
- 2026-03-30: Telegram recebeu DmPolicy minima (`open`/`allowlist`) configuravel via `CALOSUM_TELEGRAM_DM_POLICY` e `CALOSUM_TELEGRAM_ALLOWLIST`.
- 2026-03-30: `sleep_mode` agora emite evento interno `SleepModeCompletedEvent` e dispara `NightTrainer` automaticamente quando injetado pelo factory.
- 2026-03-30: Validacao concluida com `PYTHONPATH=src python3 -m calosum.harness_checks` e `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t .` (82 testes, OK).
