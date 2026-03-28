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
2. **Auto-aprendizado e Otimização de Prompts (DSPy):** Usar o backlog gerado pelo `SleepModeConsolidator` para alimentar os otimizadores do DSPy (como MIPROv2 e BootstrapFewShot) durante o Sleep Mode. Isso automatizará a melhoria do *system prompt* e a extração de *few-shots* do Hemisfério Esquerdo sem os riscos do fine-tuning puro de pesos. (Veja [Design Doc do DSPy](design-docs/dspy-self-learning.md)).
3. **Treinamento LoRA Contínuo:** Como alternativa ou complemento ao DSPy, usar o backlog para efetivamente fazer *fine-tuning* local nos pesos dos modelos (ex: via `dspy.BootstrapFinetune` ou rotinas nativas do PEFT) durante a noite.
4. **Active Inference (Free Energy):** Modificar o Hemisfério Direito para calcular a *Loss* preditiva do input do usuário. Entradas muito "surpreendentes" abaixam a temperatura do LLM e disparam mais passos de reflexão antes de agir.
