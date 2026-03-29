# Calosum Production Roadmap

Este documento lista os próximos passos estratégicos para evoluir o Calosum.

> **Status Atual (2026-03-29):** Fases 1, 2 e 3 concluídas. Fase 4 (DSPy/LoRA) com adapters implementados, pendente integração completa com sleep mode loop. Fase 5 (AI-OS Self-Awareness) em execução via plano ativo `docs/exec-plans/active/2026-03-29-ai-os-application-sprints.md`.

## Fase 1: Interface e Usabilidade (✅ Concluído)

1. ~~**Interactive REPL (CLI Chat):**~~ Implementado via `python3 -m calosum.bootstrap.cli chat`.
2. ~~**API REST / SSE:**~~ Implementado via FastAPI e Server-Sent Events (`calosum.bootstrap.api`).
3. ~~**Observabilidade (Agent UI):**~~ Painel frontend em React construído para separar a telemetria do Hemisfério Direito, Hemisfério Esquerdo e Síntese.

## Fase 2: Robustez nas Ferramentas (✅ Concluído)

1. ~~**Routing de Ações (Tools):**~~ `search_web` (via duckduckgo) e `write_file` integrados no `ConcreteActionRuntime`.
2. ~~**Sistema de Permissões:**~~ Vault de credenciais injetado via `settings.py`.
3. ~~**Cadeias de Markov e Múltiplos Passos:**~~ O loop de auto-correção via `AgentExecutionEngine` já captura erros e reprompta o modelo.

## Fase 3: Integração com o Mundo (✅ Concluído)

1. ~~**Integração de Mensageria:**~~ `TelegramChannelAdapter` implementado e exposto via `bootstrap/api.py`.
2. ~~**Expansão de Ações (Tools):**~~ `read_file`, `write_file`, `execute_bash` e `search_web` integrados no `ConcreteActionRuntime` com sandboxing e timeout.

## Fase 4: SOTA - Memória e Inferência Ativa (Prioridade Média)

1. ~~**Destilação Episódica (Neuroplasticidade):**~~ `QdrantDualMemoryAdapter` com `SleepModeConsolidator`. Consolida experiências em `SemanticRule`s.
2. ~~**Active Inference (Free Energy):**~~ Cálculo de surpresa e modulação dinâmica de temperatura no `Orchestrator` e `Bridge`. Group turns acionados quando `surprise > 0.6` ou `ambiguity > 0.8`.
3. ~~**Adapters DSPy e LoRA:**~~ `NightTrainerDSPyAdapter` (MIPROv2/BootstrapFewShot) e `NightTrainerLoRAAdapter` (PEFT/Unsloth) implementados. Pendente: integração completa com o loop de sleep mode para disparar otimização automática de prompts e fine-tuning a partir de `dspy_dataset.jsonl` e `lora_sharegpt.jsonl`.

## Fase 5: AI-OS Self-Awareness (Em Execução)

Ver plano ativo: `docs/exec-plans/active/2026-03-29-ai-os-application-sprints.md`

1. **Sprint 0 (✅):** Tipos de capacidade (`CapabilityDescriptor`, `ModelDescriptor`, `ToolDescriptor`, `ComponentHealth`) em `shared/types.py`. Builder expõe `build_capability_snapshot()` e `describe()`.
2. **Sprint 1:** Self-model e API de arquitetura (`GET /v1/system/architecture`, `GET /v1/system/capabilities`).
3. **Sprint 2:** `CognitiveWorkspace` compartilhado por turno e `GET /v1/system/state`.
4. **Sprint 3:** Introspection engine com diagnóstico de gargalos e `GET /v1/system/awareness`.
5. **Sprint 4:** Awareness loop com fila de diretivas `EvolutionDirective`. Auto-aplicação apenas para diretivas `PARAMETER`.
6. **Sprint 5:** Respostas introspectivas baseadas em estado real do sistema. Ação `introspect_self` registrada no runtime.
7. **Sprint 6:** UI com modos Arquitetura, Estado e Awareness.
8. **Sprint 7:** Hardening de routing policy e testes herméticos.
