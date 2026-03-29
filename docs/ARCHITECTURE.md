# Architecture

## Objetivo

Manter o crescimento do projeto legivel para humanos e agentes, com fronteiras pequenas e verificaveis.

## Padrao Aplicado

O projeto usa `Ports and Adapters` para fronteiras e `Builder/Abstract Factory` para compor infraestrutura.

- Em `shared/ports.py` residem nossos contratos de interfaces estáveis.
- Em `bootstrap/settings.py` residem os perfis flexíveis de execução local (incluindo o Vault de credenciais).
- Em `bootstrap/factory.py` é configurado o bootstrap do agente, injetando `adapters` no `domain`.

## Camadas

1. **`shared/`** (`types.py`, `ports.py`, `tools.py`, `async_utils.py`, `serialization.py`)
   Tipos compartilhados, contratos de dados, `ToolRegistry` e utilitários puros de serialização. Inclui os descritores de Sprint 0: `CapabilityDescriptor`, `ModelDescriptor`, `ToolDescriptor`, `ComponentHealth`.
2. **`domain/`** (`agent_execution.py`, `bridge.py`, `event_bus.py`, `left_hemisphere.py`, `memory.py`, `metacognition.py`, `multiagent.py`, `orchestrator.py`, `persistent_memory.py`, `right_hemisphere.py`, `runtime.py`, `runtime_dsl.py`, `telemetry.py`, `verifier.py`)
   Modelos de negócios do agente neuro-simbólico. Pura lógica sem detalhes I/O diretos. `agent_execution.py` concentra o loop de retry/repair; `orchestrator.py` decide entre turno simples e group turn via threshold de surpresa/ambiguidade.
3. **`adapters/`** (`action_runtime.py`, `active_inference.py`, `bridge_store.py`, `channel_telegram.py`, `knowledge_graph_nanorag.py`, `llm_failover.py`, `llm_payloads.py`, `llm_qwen.py`, `memory_qdrant.py`, `night_trainer.py`, `night_trainer_dspy.py`, `night_trainer_lora.py`, `right_hemisphere_hf.py`, `text_embeddings.py`, `tools/code_execution.py`, `tools/http_request.py`)
   Implementações concretas dos ports. Todo código que depende de SDKs externos (torch, transformers, peft, qdrant-client, dspy, openai, httpx, telegram) reside aqui exclusivamente.
4. **`bootstrap/`** (`settings.py`, `factory.py`, `cli.py`, `api.py`, `__main__.py`)
   Entrada da aplicação. `factory.py` instancia adapters, constrói `capability_snapshot` real e injeta tudo no domain via `CalosumAgentBuilder`. É o único ponto autorizado a acoplar adapters e domain.

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
- O agente tem uma entrada limpa e orquestrada pelo `orchestrator.py`. Interações isoladas da `cli.py` ou `api.py` não vazam contexto pro `domain`.

## Crescimento Controlado

- Toda nova integracao externa deve entrar em `adapters` atrás de um `Protocol`.
- A arquitetura esta mecanicamente enforcada pelo `harness_checks.py` no que diz respeito a docs obrigatorios, planos, tamanho de modulo e fronteiras de importacao. Regras novas devem virar checks mecanicos quando se tornarem recorrentes.
- Toda mudanca cross-cutting deve ter plano versionado.
