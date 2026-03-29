# Calosum Production Roadmap

Este documento lista os próximos passos estratégicos para evoluir o Calosum.

> **Status Atual:** As Fases 1, 2 e o pipeline de Destilação Episódica da Fase 4 foram amplamente implementados no sprint de `2026-03-28-aspirational-roadmap.md`. O foco agora deve se voltar para Integrações Externas e Autonomia Avançada (Active Inference).

## Fase 1: Interface e Usabilidade (✅ Concluído)

1. ~~**Interactive REPL (CLI Chat):**~~ Implementado via `python3 -m calosum.bootstrap.cli chat`.
2. ~~**API REST / SSE:**~~ Implementado via FastAPI e Server-Sent Events (`calosum.bootstrap.api`).
3. ~~**Observabilidade (Agent UI):**~~ Painel frontend em React construído para separar a telemetria do Hemisfério Direito, Hemisfério Esquerdo e Síntese.

## Fase 2: Robustez nas Ferramentas (✅ Concluído)

1. ~~**Routing de Ações (Tools):**~~ `search_web` (via duckduckgo) e `write_file` integrados no `ConcreteActionRuntime`.
2. ~~**Sistema de Permissões:**~~ Vault de credenciais injetado via `settings.py`.
3. ~~**Cadeias de Markov e Múltiplos Passos:**~~ O loop de auto-correção via `AgentExecutionEngine` já captura erros e reprompta o modelo.

## Fase 3: Integração com o Mundo (Prioridade Alta)

1. **Integração de Mensageria:** Desenvolver um script client conectando a API REST (ou SSE) ao Telegram ou WhatsApp para testes de campo.
2. **Expansão de Ações (Tools):** Implementar ações de leitura de arquivos locais e execução de scripts sandboxed, para que o agente atue ativamente sobre projetos.

## Fase 4: SOTA - Memória e Inferência Ativa (Prioridade Média)

1. ~~**Destilação Episódica (Neuroplasticidade):**~~ Implementado no `QdrantDualMemoryAdapter` integrando o `SleepModeConsolidator`. O agente consolida experiências em `SemanticRule`s no Qdrant.
2. ~~**Active Inference (Free Energy):**~~ Implementado o cálculo de Surpresa e modulação dinâmica da temperatura no `Orchestrator` e `Bridge`.
3. **Auto-aprendizado e Otimização de Prompts (DSPy):** Usar o dataset gerado pelo `SleepModeConsolidator` (`dspy_dataset.jsonl`) para alimentar os otimizadores do DSPy (como MIPROv2) durante o Sleep Mode. Isso automatizará a melhoria do *system prompt* e a extração de *few-shots*.
4. **Treinamento LoRA Contínuo:** Usar o dataset gerado (`lora_sharegpt.jsonl`) para fazer *fine-tuning* local nos pesos dos modelos durante a noite (via PEFT/Unsloth).
