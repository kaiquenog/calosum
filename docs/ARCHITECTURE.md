# Architecture

## Objetivo

Manter o crescimento do projeto legivel para humanos e agentes, com fronteiras pequenas e verificaveis.

## Padrao Aplicado

O projeto usa `Ports and Adapters` para fronteiras e `Builder/Abstract Factory` para compor infraestrutura.

- Em `shared/ports.py` residem nossos contratos de interfaces estáveis.
- Em `bootstrap/settings.py` residem os perfis flexíveis de execução local (incluindo o Vault de credenciais).
- Em `bootstrap/factory.py` é configurado o bootstrap do agente, injetando `adapters` no `domain`.

## Camadas

1. **`shared/`** (`types.py`, `ports.py`, `tools.py`, `async_utils.py`, `schemas.py`, `serialization.py`)
   Tipos compartilhados, contratos de dados, `ToolRegistry` e utilitários puros de serialização. Inclui os descritores de Sprint 0: `CapabilityDescriptor`, `ModelDescriptor`, `ToolDescriptor`, `ComponentHealth`. Ports centrais: `LeftHemispherePort`, `RightHemispherePort`, `BridgeFusionPort`, `ExperienceStorePort`, `ActionRuntimePort`, `MemorySystemPort`, `ReflectionControllerPort`, `TelemetryBusPort`.
2. **`domain/`** (`agent_config.py`, `agent_execution.py`, `bridge.py`, `differentiable_logic.py`, `directive_guardrails.py`, `event_bus.py`, `evolution.py`, `execution_utils.py`, `idle_foraging.py`, `introspection.py`, `introspection_capabilities.py`, `left_hemisphere.py`, `memory.py`, `metacognition.py`, `multiagent.py`, `orchestrator.py`, `persistent_memory.py`, `right_hemisphere.py`, `runtime.py`, `runtime_dsl.py`, `self_model.py`, `telemetry.py`, `verifier.py`, `workspace.py`)
   Modelos de negócios do agente neuro-simbólico. Pura lógica sem detalhes I/O diretos. `agent_execution.py` concentra o loop de retry/repair; `orchestrator.py` decide entre turno simples e group turn via threshold de surpresa/ambiguidade; `evolution.py` gerencia auto-evolução de parâmetros via GEA; `metacognition.py` centraliza reflexão e seleção de variantes; `introspection_capabilities.py` expõe snapshot de capacidades; `idle_foraging.py` implementa busca epistêmica passiva; `directive_guardrails.py` valida diretivas de evolução; `differentiable_logic.py` provê lógica fuzzy/LTN.
3. **`adapters/`** (`action_runtime.py`, `active_inference.py`, `bridge_cross_attention.py`, `bridge_store.py`, `channel_telegram.py`, `contract_wrappers.py`, `gea_experience_distributed.py`, `gea_experience_store.py`, `gea_reflection_experience.py`, `knowledge_graph_nanorag.py`, `latent_exchange.py`, `left_hemisphere_rlm.py`, `llm_failover.py`, `llm_payloads.py`, `llm_qwen.py`, `mcp_client.py`, `memory_qdrant.py`, `multimodal_perception.py`, `night_trainer.py`, `night_trainer_dspy.py`, `night_trainer_lora.py`, `right_hemisphere_hf.py`, `right_hemisphere_jepars.py`, `right_hemisphere_vjepa21.py`, `right_hemisphere_vljepa.py`, `telemetry_otlp.py`, `text_embeddings.py`, `tools/code_execution.py`, `tools/http_request.py`, `tools/introspection.py`, `tools/mcp_tool.py`, `tools/persistent_shell.py`, `tools/subordinate_agent.py`)
   Implementações concretas dos ports. Todo código que depende de SDKs externos (torch, transformers, peft, qdrant-client, dspy, openai, httpx, telegram) reside aqui exclusivamente. Backends do hemisférico direito: `right_hemisphere_hf.py` (HuggingFace), `right_hemisphere_vjepa21.py` (V-JEPA 2.1), `right_hemisphere_vljepa.py` (VL-JEPA multimodal), `right_hemisphere_jepars.py` (Rust/Burn JEPA-rs). Backend esquerdo alternativo: `left_hemisphere_rlm.py` (RLM recursivo). Bridge: `bridge_cross_attention.py` (fusão por cross-attention aprendida). GEA: `gea_experience_store.py`, `gea_experience_distributed.py`, `gea_reflection_experience.py`. Wrappers: `contract_wrappers.py` (enforcement de contratos por hemisferio).
4. **`bootstrap/`** (`settings.py`, `backend_resolvers.py`, `factory.py`, `context.py`, `cli.py`, `api.py`, `routers/system.py`, `routers/chat.py`, `routers/telemetry.py`, `__main__.py`)
   Entrada da aplicação. `factory.py` instancia adapters, constrói `capability_snapshot` real e injeta tudo no domain via `CalosumAgentBuilder`. `backend_resolvers.py` centraliza a lógica de seleção de backends por feature flags (hemisferio direito, esquerdo e bridge), mantendo `factory.py` desacoplado das decisões de routing. `context.py` concentra singletons/cache e resolução de settings para API. `routers/*` mantém as rotas FastAPI segmentadas por domínio HTTP. O pacote `bootstrap` segue sendo o único ponto autorizado a acoplar adapters e domain.

No dominio, `interceptors.py` formaliza o contrato de hooks cognitivos observacionais; no runtime, `action_runtime.py` emite eventos de tool execution sem quebrar isolamento de camadas.

## Governanca de Harness

Fora das quatro camadas principais, o repositório mantém `harness_checks.py` na raiz do pacote `calosum` como utilitário de governança. Ele não faz parte do runtime do agente; sua função é validar artefatos obrigatórios, planos, limites de módulo (<400 linhas) e fronteiras de importação via AST.

Todo novo módulo Python deve ser registrado em `MODULE_RULES` dentro de `harness_checks.py` com o conjunto explícito de imports internos permitidos. Módulos sem regra geram violação `missing_module_rule` que quebra o build. Consulte `docs/references/harness-engineering.md` para a lista completa de checks e seus códigos de erro.

## Interface de Usuário (UI)

O projeto também possui um componente frontend na pasta `ui/` construído com React, Vite e Tailwind. Este painel consome as rotas expostas em `bootstrap/api.py` para exibir a telemetria separada por hemisférios, execução e reflexão.

## Regras

- Pacote `shared` não depende de outros pacotes internos. Serve como base de comunicação de dicionários, data classes e portas (`Protocols`).
- Pacote `domain` define o core. Ele NUNCA deve tentar importar bibliotecas SDK de "adapters" nem as instâncias do "bootstrap".
- Pacote `adapters` obedece cegamente a interface em `shared`. Não toma decisões fora de traduzir a infra.
- Pacote `bootstrap` é o único capaz e autorizado a instanciar `adapters` concretos injetando-os nas instâncias do `domain` de acordo com configs do painel `settings.py`.
- Quando um adapter opcional de infraestrutura local não estiver disponível, o `bootstrap` deve preferir fallback explícito a falha dura sempre que isso não quebrar o contrato funcional.
- Dependências opcionais de canais externos (ex.: Telegram) devem degradar graciosamente no bootstrap, preservando a disponibilidade da API principal.
- O agente tem uma entrada limpa e orquestrada pelo `orchestrator.py`. Interações isoladas da `cli.py` ou `api.py` não vazam contexto pro `domain`.

## Crescimento Controlado

- Toda nova integracao externa deve entrar em `adapters` atrás de um `Protocol`.
- A arquitetura esta mecanicamente enforcada pelo `harness_checks.py` no que diz respeito a docs obrigatorios, planos, tamanho de modulo e fronteiras de importacao. Regras novas devem virar checks mecanicos quando se tornarem recorrentes.
- Toda mudanca cross-cutting deve ter plano versionado.
