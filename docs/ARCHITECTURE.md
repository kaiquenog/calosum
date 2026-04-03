# Architecture

## Objetivo

Manter o crescimento do projeto legivel para humanos e agentes, com fronteiras pequenas e verificaveis.

## Padrao Aplicado

O projeto usa `Ports and Adapters` para fronteiras e `Builder/Abstract Factory` para compor infraestrutura.

- Em `shared/models/ports.py` residem nossos contratos de interfaces estáveis.
- Em `bootstrap/infrastructure/settings.py` residem os perfis flexíveis de execução local (incluindo o Vault de credenciais).
- Em `bootstrap/wiring/factory.py` é configurado o bootstrap do agente, injetando `adapters` no `domain`.

## Camadas

1. **`shared/`** (`models/types.py`, `models/ports.py`, `models/schemas.py`, `utils/async_utils.py`, `utils/serialization.py`, `utils/free_energy.py`, `utils/surprise_metrics.py`)
   Tipos compartilhados, contratos de dados, `ToolRegistry` e utilitários puros de serialização. Inclui os descritores de Sprint 0: `CapabilityDescriptor`, `ModelDescriptor`, `ToolDescriptor`, `ComponentHealth`. Ports centrais: `LeftHemispherePort`, `RightHemispherePort`, `BridgeFusionPort`, `ExperienceStorePort`, `ActionRuntimePort`, `MemorySystemPort`, `ReflectionControllerPort`, `TelemetryBusPort`.
2. **`domain/`** (subfolders: `agent/`, `cognition/`, `execution/`, `infrastructure/`, `memory/`, `metacognition/`)
   Modelos de negócios do agente neuro-simbólico. Pura lógica sem detalhes I/O diretos.
   - **`agent/`**: `orchestrator.py` (coordenação do loop cognitivo), `orchestrator_briefing.py`, `orchestrator_utils.py`, `evolution.py` (evolução via GEA), `idle_foraging.py` (busca epistêmica passiva), `multiagent.py` (comunicação entre agentes), `agent_config.py`, `directive_guardrails.py`.
   - **`cognition/`**: `action_planner.py` (planejamento estruturado), `input_perception.py` (percepção neural/heurística), `bridge.py` (fusão de hemisférios), `differentiable_logic.py` (lógica fuzzy/LTN).
   - **`execution/`**: `agent_execution.py` (retry/repair), `tool_runtime.py` (instância de ferramentas), `workspace.py` (contexto operacional), `execution_utils.py`.
   - **`infrastructure/`**: `event_bus.py`, `telemetry.py`, `verifier.py`, `interceptors.py` (hooks observacionais).
   - **`memory/`**: `memory.py` (lógica de memória episódica).
   - **`metacognition/`**: `metacognition.py` (reflexão central), `self_model.py` (auto-representação), `awareness.py`, `introspection.py` (diagnóstico interno), `introspection_capabilities.py` (snapshot de capacidades).
3. **`adapters/`** (subfolders: `llm/`, `memory/`, `hemisphere/`, `tools/`, `perception/`, `bridge/`, `communication/`, `experience/`, `night_trainer/`, `knowledge/`)
   Implementações concretas dos ports. Todo código que depende de SDKs externos (torch, transformers, peft, qdrant-client, dspy, openai, httpx, telegram) reside aqui exclusivamente.
   - **`llm/`**: `llm_qwen.py`, `llm_failover.py`, `llm_fusion.py`, `llm_payloads.py`, `llm_payload_parser.py` (extraído para conformidade de harness).
   - **`memory/`**: `memory_qdrant.py` (vetores latentes), `text_embeddings.py` (vectorizers).
   - **`hemisphere/`**: `input_perception_hf.py`, `input_perception_heuristic_jepa.py`, `input_perception_trained_jepa.py`, `input_perception_vjepa21.py`, `input_perception_vljepa.py`, `input_perception_jepars.py` (Rust/Burn), `left_hemisphere_rlm_ast.py`.
   - **`tools/`**: `persistent_shell.py`, `http_request.py`, `introspection.py`, `mcp_tool.py`, `mcp_client.py`, `subordinate_agent.py`.
   - **`perception/`**: `multimodal_perception.py`, `quantized_embeddings.py` (TurboQuant), `simple_distance.py`.
   - **`communication/`**: `channel_telegram.py`, `telemetry_otlp.py`.
   - **`experience/`**: `gea_experience_store.py`, `gea_experience_distributed.py`, `gea_reflection_experience.py`, `variant_preference.py`.
   - **`night_trainer/`**: `night_trainer.py`, `night_trainer_dspy.py`.
   - **`infrastructure/`**: `contract_wrappers.py`.
4. **`bootstrap/`** (`infrastructure/settings.py`, `wiring/backend_resolvers.py`, `wiring/factory.py`, `wiring/agent_baseline.py`, `entry/context.py`, `entry/cli.py`, `entry/api.py`, `routers/system.py`, `routers/chat.py`, `routers/telemetry.py`, `entry/__main__.py`)
   Entrada da aplicação. `wiring/factory.py` instancia adapters, constrói `capability_snapshot` real e injeta tudo no domain via `CalosumAgentBuilder`. `wiring/backend_resolvers.py` centraliza a lógica de seleção de backends por feature flags (hemisferio direito, esquerdo e bridge), mantendo `wiring/factory.py` desacoplado das decisões de routing. `wiring/agent_baseline.py` provê o baseline versionado usado pelos gates de benchmark. `entry/context.py` concentra singletons/cache e resolução de settings para API. `routers/*` mantém as rotas FastAPI segmentadas por domínio HTTP. O pacote `bootstrap` segue sendo o único ponto autorizado a acoplar adapters e domain.

No dominio, `interceptors.py` formaliza o contrato de hooks cognitivos observacionais; no runtime, `domain/execution/tool_runtime.py` mantém a execução tipificada isolada das integrações concretas.

## Governanca de Harness

Fora das quatro camadas principais, o repositório mantém `harness_checks.py` na raiz do pacote `calosum` como utilitário de governança. Ele não faz parte do runtime do agente; sua função é validar artefatos obrigatórios, planos, limites de módulo (<400 linhas) e fronteiras de importação via AST.

Todo novo módulo Python deve ser registrado em `MODULE_RULES` dentro de `harness_checks.py`. O sistema de testes também segue esta estrutura modular em `tests/`, dividido em `tests/domain/`, `tests/adapters/`, `tests/bootstrap/`, `tests/shared/` e `tests/integration/`. Além de fronteiras e tamanho de módulo, o harness valida docstrings de pacote semântico, ausência de imports indevidos de `domain` em `shared/` (fora de `TYPE_CHECKING`), isolamento de adapters contra `os`/`subprocess` diretos e padrões proibidos de ML/treino em `domain/`. Consulte `docs/references/harness-engineering.md` para a lista completa de códigos de erro e a ordem de execução.


## CI/CD e Gates de Qualidade

O workflow em `.github/workflows/ci.yml` executa quatro jobs em cadeia (`needs`):

1. `lint_types`: `harness_checks`, `ruff` e `mypy --strict` focados em arquivos Python alterados.
2. `unit_tests`: suite unitária + cobertura com gate de 80% para modulos Python novos/alterados em `src/calosum/`.
3. `integration`: benchmark de integracao com gate de latencia `p95 <= 5000 ms` em perfil `ephemeral`.
4. `benchmark_gate`: comparacao automatica contra baseline versionado, falhando se houver regressao > 5% em `tool_success_rate`.

Os artefatos gerados em cada run ficam em `docs/benchmarks/ci/` e sao publicados como artifact do CI.

## Componentes Renomeados no Sprint 0

- `ContextCompressor` (alias legado `CognitiveTokenizer`): `docs/components/context-compressor/ARCHITECTURE.md`
- `CognitiveVariantSelector` (alias legado `GEAReflectionController`): `docs/components/cognitive-variant-selector/ARCHITECTURE.md`
- Loop de adaptacao (`apply_neuroplasticity`): `docs/components/neuroplasticity-loop/ARCHITECTURE.md`

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
