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
2. **`domain/`** (subfolders: `agent/`, `cognition/`, `execution/`, `infrastructure/`, `memory/`, `metacognition/`)
   Modelos de negócios do agente neuro-simbólico. Pura lógica sem detalhes I/O diretos.
   - **`agent/`**: `orchestrator.py` (coordenação do loop cognitivo), `evolution.py` (evolução via GEA), `idle_foraging.py` (busca epistêmica passiva), `multiagent.py` (comunicação entre agentes), `agent_config.py`, `directive_guardrails.py`.
   - **`cognition/`**: `left_hemisphere.py` (lógica simbólica/RLM), `right_hemisphere.py` (percepção neural), `bridge.py` (fusão de hemisférios), `differentiable_logic.py` (lógica fuzzy/LTN).
   - **`execution/`**: `agent_execution.py` (retry/repair), `runtime.py` (instância de ferramentas), `workspace.py` (contexto operacional), `execution_utils.py`, `runtime_dsl.py`, `group_turn.py`.
   - **`infrastructure/`**: `event_bus.py`, `telemetry.py`, `verifier.py`, `interceptors.py` (hooks observacionais).
   - **`memory/`**: `memory.py` (lógica de memória episódica), `persistent_memory.py`.
   - **`metacognition/`**: `metacognition.py` (reflexão central), `self_model.py` (auto-representação), `awareness.py`, `introspection.py` (diagnóstico interno), `introspection_capabilities.py` (snapshot de capacidades).
3. **`adapters/`** (subfolders: `llm/`, `memory/`, `hemisphere/`, `tools/`, `perception/`, `bridge/`, `communication/`, `experience/`, `night_trainer/`, `knowledge/`)
   Implementações concretas dos ports. Todo código que depende de SDKs externos (torch, transformers, peft, qdrant-client, dspy, openai, httpx, telegram) reside aqui exclusivamente.
   - **`llm/`**: `llm_qwen.py`, `llm_failover.py`, `llm_payload_parser.py` (extraído para conformidade de harness).
   - **`memory/`**: `memory_qdrant.py` (vetores latentes), `text_embeddings.py` (vectorizers).
   - **`hemisphere/`**: `right_hemisphere_hf.py`, `right_hemisphere_vjepa21.py`, `right_hemisphere_vljepa.py`, `right_hemisphere_jepars.py` (Rust/Burn), `left_hemisphere_rlm.py`.
   - **`tools/`**: `code_execution.py`, `persistent_shell.py`, `mcp_tool.py`, `mcp_client.py`, `subordinate_agent.py`.
   - **`perception/`**: `active_inference.py`, `multimodal_perception.py`, `quantized_embeddings.py` (TurboQuant).
   - **`communication/`**: `channel_telegram.py`, `telemetry_otlp.py`.
   - **`experience/`**: `gea_experience_store.py`, `gea_experience_distributed.py`.
   - **`night_trainer/`**: `night_trainer.py`, `night_trainer_dspy.py`.
   - **`execution/`**: `action_runtime.py`.
   - **`infrastructure/`**: `contract_wrappers.py`.
4. **`bootstrap/`** (`settings.py`, `backend_resolvers.py`, `factory.py`, `context.py`, `cli.py`, `api.py`, `routers/system.py`, `routers/chat.py`, `routers/telemetry.py`, `__main__.py`)
   Entrada da aplicação. `factory.py` instancia adapters, constrói `capability_snapshot` real e injeta tudo no domain via `CalosumAgentBuilder`. `backend_resolvers.py` centraliza a lógica de seleção de backends por feature flags (hemisferio direito, esquerdo e bridge), mantendo `factory.py` desacoplado das decisões de routing. `context.py` concentra singletons/cache e resolução de settings para API. `routers/*` mantém as rotas FastAPI segmentadas por domínio HTTP. O pacote `bootstrap` segue sendo o único ponto autorizado a acoplar adapters e domain.

No dominio, `interceptors.py` formaliza o contrato de hooks cognitivos observacionais; no runtime, `action_runtime.py` emite eventos de tool execution sem quebrar isolamento de camadas.

## Governanca de Harness

Fora das quatro camadas principais, o repositório mantém `harness_checks.py` na raiz do pacote `calosum` como utilitário de governança. Ele não faz parte do runtime do agente; sua função é validar artefatos obrigatórios, planos, limites de módulo (<400 linhas) e fronteiras de importação via AST.

Todo novo módulo Python deve ser registrado em `MODULE_RULES` dentro de `harness_checks.py`. O sistema de testes também segue esta estrutura modular em `tests/`, dividido em `tests/domain/`, `tests/adapters/`, `tests/bootstrap/`, `tests/shared/` e `tests/integration/`. Consulte `docs/references/harness-engineering.md` para a lista completa de checks e seus códigos de erro.

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
