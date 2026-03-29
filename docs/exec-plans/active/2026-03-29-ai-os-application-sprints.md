# Title
Aplicacao de AI Operating System no Calosum em Sprints

## Purpose
Transformar a analise de encaixe `docs/reports/2026-03-29-ai-os-fit-analysis.md` em um plano de execucao aplicavel, incremental e validavel. O objetivo nao e substituir a arquitetura atual por um framework externo, mas evoluir o Calosum para um runtime mais autoconsciente com:

- self-model de runtime;
- capability host explicito;
- workspace cognitivo compartilhado;
- awareness loop;
- interface humana e interna mais consciente das capacidades e do estado do sistema.

Este plano e separado do plano `2026-03-29-self-awareness-evolution.md` porque aquele documento descreve a evolucao conceitual do nucleo de autoconsciencia, enquanto este documento organiza a **aplicacao pratica por sprints**, com ordem, dependencias, gates e entregas.

## Scope

### In Scope

- Promover `builder.describe()` para um self-model/control plane reutilizavel.
- Expor estado arquitetural, capacidades e health do sistema via API.
- Introduzir um `CognitiveWorkspace` compartilhado por turno.
- Enriquecer telemetria com dados de backends, modelos, tools e awareness.
- Implementar introspection e diagnostico de gargalos.
- Implementar fila de diretivas e awareness loop com auto-aplicacao apenas para mudancas parametricas.
- Permitir respostas introspectivas baseadas em dados reais.
- Evoluir a UI para exibir arquitetura, estado e awareness.

### Out of Scope

- Substituir o Calosum por AIOS, Letta, MemGPT ou outro framework.
- Implementar MCP server completo neste ciclo.
- Federacao multi-instancia ou multiagente distribuido.
- Auto-modificacao arquitetural sem aprovacao humana.
- Troca massiva de backends/modelos como eixo principal do trabalho.

## Sprint Assumptions

- Sprints sequenciais, com acoplamento controlado.
- Cada sprint fecha com artefato executavel, docs atualizadas e gates mecanicos.
- Qualquer mudanca cross-cutting deve preservar `Ports and Adapters`.
- Cada sprint deve manter fallbacks explicitos e evitar dependencia dura de servicos externos.
- Comandos padrao de validacao por sprint:
  - `PYTHONPATH=src python3 -m calosum.harness_checks`
  - `PYTHONPATH=src python3 -m unittest discover -s tests -t .`

## Validation

### Gates Globais

- O runtime continua processando turnos simples e group turns sem regressao funcional.
- O agente consegue expor sua arquitetura e capacidades sem consultar texto hardcoded.
- O painel UI mostra nao apenas timeline, mas tambem arquitetura, estado e awareness.
- Awareness gera diagnosticos coerentes a partir de telemetria historica.
- Apenas diretivas `PARAMETER` sao auto-aplicadas; `PROMPT`, `TOPOLOGY` e `ARCHITECTURE` ficam pendentes.

### Gates por Sprint

- Cada sprint adiciona testes direcionados para os contratos novos.
- Cada endpoint novo tem teste de API.
- Cada tipo novo em `shared/` e coberto por serializacao/consumo minimo.
- Toda nova superficie introspectiva e derivada de estado real do runtime.

## Sprint Plan

### Sprint 0 - Baseline de Capability State e Contratos de Observabilidade

**Objetivo:** Criar a base contratual para que o sistema saiba e publique quais backends, modelos, tools e capacidades estao ativos agora.

**Touch points principais:**

- `src/calosum/bootstrap/factory.py`
- `src/calosum/bootstrap/api.py`
- `src/calosum/domain/agent_execution.py`
- `src/calosum/domain/telemetry.py`
- `src/calosum/adapters/action_runtime.py`
- `src/calosum/shared/types.py`
- `src/calosum/shared/tools.py`
- `tests/`

**Entregas:**

- Introduzir tipos de snapshot para capacidades do sistema:
  - `CapabilityDescriptor`
  - `ModelDescriptor`
  - `ToolDescriptor`
  - `ComponentHealth`
- Extrair do runtime um snapshot minimo reutilizavel com:
  - backend do hemisferio direito;
  - backend/modelo/provedor do hemisferio esquerdo;
  - embedding backend;
  - knowledge graph backend;
  - tools registradas;
  - permissoes requeridas;
  - estado `healthy/degraded/unavailable`.
- Enriquecer a telemetria de turno com referencias a:
  - modelo/backend do hemisferio direito;
  - modelo/backend do hemisferio esquerdo;
  - variante vencedora;
  - config snapshot do bridge;
  - registry snapshot das tools.

**Validacao do sprint:**

- `builder.describe()` deixa de ser apenas CLI helper e passa a ser schema reutilizavel.
- Dashboard e payloads carregam backend/modelo sem depender de parsing textual.
- Testes cobrem serializacao do snapshot e telemetria enriquecida.

**Gate de saida:**

- O sistema consegue responder programaticamente "quais capacidades estao ativas?" sem inspecao manual de config.

### Sprint 1 - Self-Model e Architecture API

**Objetivo:** Introduzir um mapa interno explicito da arquitetura executando agora.

**Touch points principais:**

- `src/calosum/domain/self_model.py` novo
- `src/calosum/domain/orchestrator.py`
- `src/calosum/shared/types.py`
- `src/calosum/bootstrap/api.py`
- `tests/`

**Entregas:**

- Implementar `CognitiveArchitectureMap` com:
  - componentes;
  - conexoes;
  - adaptation surface;
  - capabilities anexadas;
  - health/status por componente.
- Implementar `build_self_model(agent)` usando reflection sobre o agente instanciado.
- Persistir `self_model` em memoria no boot do `CalosumAgent`.
- Expor:
  - `GET /v1/system/architecture`
  - `GET /v1/system/capabilities`

**Validacao do sprint:**

- O endpoint retorna qual adapter implementa cada hemisferio.
- O endpoint retorna configuracoes ajustaveis do bridge.
- O endpoint lista tools e permissoes registradas no runtime.
- Testes garantem schema minimo em perfis `ephemeral` e `persistent`.

**Gate de saida:**

- O runtime possui self-model somente leitura, consumivel por API, UI e futuro raciocinio introspectivo.

### Sprint 2 - Shared Cognitive Workspace por Turno

**Objetivo:** Criar a interface que hoje falta entre os hemisferios, bridge, verifier e runtime.

**Touch points principais:**

- `src/calosum/domain/workspace.py` novo
- `src/calosum/domain/orchestrator.py`
- `src/calosum/domain/agent_execution.py`
- `src/calosum/domain/bridge.py`
- `src/calosum/domain/left_hemisphere.py`
- `src/calosum/domain/verifier.py`
- `src/calosum/adapters/action_runtime.py`
- `src/calosum/shared/types.py`
- `tests/`

**Entregas:**

- Introduzir `CognitiveWorkspace` com campos como:
  - `task_frame`
  - `self_model_ref`
  - `capability_snapshot`
  - `right_notes`
  - `bridge_state`
  - `left_notes`
  - `verifier_feedback`
  - `runtime_feedback`
  - `pending_questions`
- Propagar o workspace ao longo do pipeline de turno.
- Fazer cada componente escrever apenas a sua secao.
- Expor `GET /v1/system/state` para o workspace do ultimo turno por sessao.

**Validacao do sprint:**

- Um turno completo pode ser reconstituido como estado compartilhado, nao apenas como timeline.
- O runtime informa claramente quando houve:
  - tool indisponivel;
  - permissao faltando;
  - mismatch entre acao proposta e vocabulario aceito;
  - feedback corretivo do verifier.
- Testes garantem que o workspace e preenchido incrementalmente sem quebrar o contrato dos ports.

**Gate de saida:**

- A interacao entre hemisferios deixa de ser apenas `RightHemisphereState -> CognitiveBridgePacket -> LeftHemisphereResult` e passa a ter memoria operacional de turno.

### Sprint 3 - Introspection Engine e Canal Awareness

**Objetivo:** Converter telemetria historica em diagnostico utilizavel.

**Touch points principais:**

- `src/calosum/domain/introspection.py` novo
- `src/calosum/domain/telemetry.py`
- `src/calosum/domain/orchestrator.py`
- `src/calosum/shared/types.py`
- `src/calosum/bootstrap/api.py`
- `tests/`

**Entregas:**

- Implementar `IntrospectionEngine` para agregar:
  - taxa de sucesso de tools;
  - retry rate;
  - tipos de falha;
  - dominancia de variantes;
  - tendencia de surprise;
  - backlog de aprovacoes;
  - sinais de dessensibilizacao do hemisferio direito.
- Implementar `SessionDiagnostic` e `CognitiveBottleneck`.
- Adicionar canal `awareness` ao barramento de telemetria.
- Expor `GET /v1/system/awareness`.

**Validacao do sprint:**

- O sistema identifica gargalos recorrentes a partir de traces reais.
- O dashboard consegue mostrar bottlenecks em vez de apenas eventos brutos.
- Testes sinteticos cobrem cenarios como:
  - retry rate alto;
  - variante empatica nunca vence;
  - tool not found recorrente;
  - surprise sempre baixo.

**Gate de saida:**

- O agente possui diagnostico operacional real, nao apenas observabilidade passiva.

### Sprint 4 - Awareness Loop e Fila de Diretivas

**Objetivo:** Fechar o ciclo entre diagnostico e acao controlada.

**Touch points principais:**

- `src/calosum/domain/evolution.py` novo
- `src/calosum/domain/orchestrator.py`
- `src/calosum/domain/bridge.py`
- `src/calosum/shared/types.py`
- `src/calosum/bootstrap/api.py`
- `tests/`

**Entregas:**

- Implementar `EvolutionDirective` e fila de diretivas pendentes.
- Implementar `EvolutionProposer`.
- Adicionar awareness loop no `CalosumAgent` com frequencia configuravel.
- Auto-aplicar apenas diretivas `PARAMETER`.
- Persistir historico minimo em:
  - `.calosum-runtime/evolution/archive.jsonl`
- Expor:
  - `GET /v1/system/directives`
  - `POST /v1/system/directives/apply` para aprovacoes explicitas futuras.

**Validacao do sprint:**

- Parametros do bridge podem ser ajustados automaticamente com evidencia observavel.
- Diretivas de escopo maior ficam pendentes e explicitadas.
- Testes garantem que nao ha auto-aplicacao de `PROMPT`, `TOPOLOGY` ou `ARCHITECTURE`.

**Gate de saida:**

- O runtime consegue propor e aplicar pequenas evolucoes sem colapsar fronteiras arquiteturais.

### Sprint 5 - Self-Awareness Conversacional e Acao `introspect_self`

**Objetivo:** Fazer o agente falar sobre si com base em estado real do sistema.

**Touch points principais:**

- `src/calosum/domain/left_hemisphere.py`
- `src/calosum/adapters/action_runtime.py`
- `src/calosum/shared/tools.py`
- `src/calosum/shared/types.py`
- `src/calosum/bootstrap/api.py`
- `tests/`

**Entregas:**

- Adicionar deteccao de perguntas introspectivas no hemisferio esquerdo.
- Injetar `self_model`, `latest_awareness` e `workspace` no contexto de resposta introspectiva.
- Registrar nova acao `introspect_self` no runtime.
- Expor `POST /v1/system/introspect` como caminho explicito para consumo pela UI.

**Validacao do sprint:**

- O agente responde perguntas como:
  - "como voce funciona?"
  - "quais tools voce tem?"
  - "onde voce esta falhando?"
  - "o que voce sugere mudar?"
- As respostas usam dados do runtime e nao texto generico.
- Testes cobrem requests introspectivos e payload da nova action.

**Gate de saida:**

- O sistema passa a ter autoconsciencia operacional conversacional.

### Sprint 6 - UI Consciente: Architecture, State e Awareness

**Objetivo:** Evoluir a interface humana para refletir a nova superficie introspectiva do sistema.

**Touch points principais:**

- `ui/src/App.tsx`
- `ui/src/App.css`
- `ui/src/index.css`
- `src/calosum/bootstrap/api.py`
- `tests/` para API e, se existir infraestrutura, testes de UI

**Entregas:**

- Adicionar tres modos novos na UI:
  - `Architecture`
  - `State`
  - `Awareness`
- Consumir os novos endpoints do sistema.
- Exibir:
  - topologia viva dos hemisferios e bridge;
  - modelos/backends ativos;
  - tools/permissoes;
  - workspace do ultimo turno;
  - bottlenecks e diretivas.
- Reduzir dependencia de polling cego sempre que possivel, reaproveitando SSE ou atualizacoes dirigidas.

**Validacao do sprint:**

- Um operador humano consegue responder pela UI:
  - o que esta rodando agora;
  - o que o sistema sabe fazer;
  - em que estado cognitivo o ultimo turno terminou;
  - quais gargalos estao ativos;
  - quais diretivas estao pendentes.

**Gate de saida:**

- A UI deixa de ser so timeline de eventos e vira interface consciente do sistema.

### Sprint 7 - Routing Policy e Hardening de Capability Host

**Objetivo:** Tornar explicito o roteamento entre modelos/capacidades e fechar riscos operacionais identificados durante os sprints anteriores.

**Touch points principais:**

- `src/calosum/bootstrap/factory.py`
- `src/calosum/domain/orchestrator.py`
- `src/calosum/shared/types.py`
- `src/calosum/bootstrap/settings.py`
- `src/calosum/adapters/right_hemisphere_hf.py`
- `tests/`

**Entregas:**

- Introduzir policy explicita para:
  - `perception_model`
  - `reason_model`
  - `reflection_model`
  - `verifier_model` opcional
- Expor fallback e degradacao como estado de capability.
- Tornar o backend HF observavelmente `degraded/unavailable` em vez de implicitamente falhar.
- Fechar testes que hoje dependem de rede/model download com mocks ou fixtures locais.

**Validacao do sprint:**

- O sistema distingue availability de policy de roteamento.
- O estado dos modelos aparece com health consistente.
- A suite de testes deixa de depender de acesso externo para passar.

**Gate de saida:**

- O capability host fica operacionalmente previsivel, e nao apenas configuravel.

## Dependencies and Sequencing Notes

- Sprint 0 e pre-requisito para todos os demais.
- Sprint 1 pode ser iniciado assim que o snapshot de capabilities existir.
- Sprint 2 deve vir antes da awareness, porque awareness precisa de estado de turno melhor estruturado.
- Sprint 3 e 4 sao o miolo do control plane.
- Sprint 5 depende diretamente de 1, 2 e 3.
- Sprint 6 depende de 1, 2 e 3 para ter superfice real para exibir.
- Sprint 7 pode rodar em paralelo parcial com Sprint 6, desde que ownership de arquivos seja isolado.

## Progress

- [x] Sprint 0 - Baseline de Capability State e Contratos de Observabilidade
- [x] Sprint 1 - Self-Model e Architecture API
- [x] Sprint 2 - Shared Cognitive Workspace por Turno
- [ ] Sprint 3 - Introspection Engine e Canal Awareness
- [ ] Sprint 4 - Awareness Loop e Fila de Diretivas
- [ ] Sprint 5 - Self-Awareness Conversacional e Acao `introspect_self`
- [ ] Sprint 6 - UI Consciente: Architecture, State e Awareness
- [ ] Sprint 7 - Routing Policy e Hardening de Capability Host

## Decision Log

- 2026-03-29: Plano criado a partir do report `2026-03-29-ai-os-fit-analysis.md`.
- 2026-03-29: Decidido manter o plano de `self-awareness-evolution` como documento conceitual e criar este plano separado como trilha executiva em sprints.
- 2026-03-29: Decidido nao adotar AIOS/Letta/MemGPT como framework base; o trabalho sera aplicado sobre a arquitetura atual.
- 2026-03-29: Decidido que o control plane comeca por self-model + capability host antes de awareness loop.
- 2026-03-29: Decidido que apenas mudancas `PARAMETER` podem ser auto-aplicadas neste ciclo.
- 2026-03-29: Decidido reservar um sprint especifico para hardening de roteamento e testes hermeticos, porque o estado atual ainda expõe dependencia de backend externo no hemisferio direito.
