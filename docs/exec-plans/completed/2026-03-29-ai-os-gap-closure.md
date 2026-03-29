# Title
Fechamento de Gaps do AI OS Aplicado ao Calosum

## Purpose
Fechar os gaps identificados na revisao de cobertura do plano `docs/exec-plans/completed/2026-03-29-ai-os-application-sprints.md`, transformando superficies hoje descritivas ou parciais em comportamento operacional verificavel. O objetivo deste plano nao e abrir uma nova frente conceitual, mas concluir corretamente o que o ciclo anterior prometeu para:

- routing policy operacional;
- self-model e capability surfaces coerentes com o estado real;
- awareness loop com persistencia e fila de diretivas real;
- introspecao conversacional grounded em runtime state;
- UI consciente com dados completos;
- suite hermetica, sem dependencia de rede nem escrita fora dos diretorios de teste.

## Scope

### In Scope

- Tornar `RoutingPolicy` efetiva no bootstrap e no roteamento observavel do runtime.
- Corrigir inconsistencias entre `builder.describe()`, `capability_snapshot` e `self_model`.
- Completar awareness loop com frequencia configuravel, persistencia e fila de diretivas auditavel.
- Ampliar `IntrospectionEngine` para cobrir os sinais prometidos no plano anterior.
- Reestruturar `introspect_self` para responder com dados estruturados do runtime.
- Completar UI para exibir modelos, backends, permissoes, routing policy e campos faltantes do workspace.
- Fechar os pontos que hoje impedem validacao hermetica em CLI, API e testes.

### Out of Scope

- Nova arquitetura conceitual alem do que ja foi aprovado.
- MCP server, federacao multiagente ou auto-modificacao arquitetural.
- Reescrever a UI do zero.
- Troca massiva de provedores/modelos como objetivo principal.

## Validation

### Gates Globais

- `RoutingPolicy` altera comportamento observavel do runtime ou do bootstrap, nao apenas o snapshot.
- `self_model`, `/v1/system/info`, `/v1/system/capabilities` e `/v1/system/architecture` concordam entre si sobre health, tools e backends ativos.
- Awareness gera diagnosticos com sinais reais de falha, backlog de aprovacao, tendencia de surprise e dominancia de variantes.
- Apenas `PARAMETER` continua auto-aplicavel, mas diretivas maiores passam a existir e ficam auditavelmente pendentes.
- `introspect_self` responde usando `self_model`, `latest_awareness` e `workspace`, sem frases hardcoded como fonte principal.
- A UI permite responder visualmente quais modelos/backends estao ativos, quais permissoes/tools existem, como terminou o ultimo turno e quais diretivas estao pendentes.
- `PYTHONPATH=src .venv/bin/python -m calosum.harness_checks` passa.
- `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t .` passa sem depender de rede externa.

### Gates por Frente

- Cada endpoint alterado recebe teste de API.
- Toda nova persistencia tem teste de leitura/escrita minima.
- Toda surface introspectiva deriva de estado real do agente ou telemetria persistida.
- Qualquer caminho de arquivo usado por CLI, bridge ou telemetry aceita configuracao ou isolamento por teste.

## Plan

### Frente 1 - Routing Policy Operacional e Capability Health Coerente

**Objetivo:** Fazer a policy influenciar composicao e observabilidade do runtime.

**Touch points principais:**

- `src/calosum/bootstrap/settings.py`
- `src/calosum/bootstrap/factory.py`
- `src/calosum/domain/self_model.py`
- `src/calosum/bootstrap/api.py`
- `src/calosum/shared/types.py`
- `tests/test_factory.py`
- `tests/test_api.py`
- `tests/test_self_model.py`

**Entregas:**

- Tornar `perception_model`, `reason_model`, `reflection_model` e `verifier_model` operacionalmente relevantes.
- Fazer `/v1/system/info` publicar tools/permissoes reais do runtime, nao snapshot vazio do builder.
- Propagar health degradado ou indisponivel do capability host para o `self_model`.
- Distinguir explicitamente policy de roteamento de availability.

### Frente 2 - Awareness Loop Real, Persistencia e Diretivas Auditaveis

**Objetivo:** Fechar o control plane que ficou parcial.

**Touch points principais:**

- `src/calosum/domain/introspection.py`
- `src/calosum/domain/evolution.py`
- `src/calosum/domain/orchestrator.py`
- `src/calosum/domain/telemetry.py`
- `src/calosum/bootstrap/api.py`
- `tests/`

**Entregas:**

- Adicionar frequencia configuravel para awareness loop.
- Persistir diagnosticos e diretivas em `.calosum-runtime/evolution/archive.jsonl` ou caminho configuravel equivalente.
- Fazer `EvolutionProposer` gerar tambem diretivas nao parametricas quando houver evidencia.
- Garantir que apenas `PARAMETER` e auto-aplicada e que os demais tipos entram em fila persistida.
- Cobrir backlog de aprovacoes, tipos de falha e tendencia de surprise no `IntrospectionEngine`.

### Frente 3 - Introspecao Grounded e UI Completa

**Objetivo:** Fazer introspecao e interface refletirem o estado real do runtime.

**Touch points principais:**

- `src/calosum/adapters/action_runtime.py`
- `src/calosum/adapters/llm_qwen.py`
- `src/calosum/domain/workspace.py`
- `src/calosum/bootstrap/api.py`
- `ui/src/App.tsx`
- `ui/src/App.css`
- `tests/test_api.py`

**Entregas:**

- Injetar `self_model`, `latest_awareness` e `workspace` em respostas introspectivas estruturadas.
- Expor no workspace e na UI os campos hoje omitidos:
  - `self_model_ref`
  - `capability_snapshot`
  - `pending_questions`
  - routing policy
  - permissoes de tools
- Reduzir polling cego quando houver surface dirigida suficiente.
- Garantir que perguntas introspectivas retornem estado real e recomendacoes baseadas em evidencias.

### Frente 4 - Hermeticidade de CLI, API e Testes

**Objetivo:** Remover dependencias implícitas de rede, interpreter externo e caminhos fixos.

**Touch points principais:**

- `src/calosum/__init__.py`
- `src/calosum/bootstrap/factory.py`
- `src/calosum/adapters/bridge_store.py`
- `src/calosum/domain/persistent_memory.py`
- `tests/test_cli.py`
- `tests/test_api.py`
- `tests/test_right_hemisphere_hf.py`

**Entregas:**

- Eliminar imports precoces que explodem quando dependencias opcionais nao estao carregadas.
- Permitir configurar path de `LocalBridgeStateStore` para testes e CLI.
- Garantir que testes de API e CLI nao escrevam em `.calosum-runtime` por default.
- Fechar testes que ainda tentam baixar modelo ou depender de rede com mocks/fixtures locais.

## Completion Summary

- A routing policy agora altera o bootstrap: `reason_model` passa a dirigir o hemisferio esquerdo, `perception_model=jepa` força a rota heuristica, `/v1/system/info` usa o agent real e publica `routing_resolution` junto de tools/permissoes concretas.
- Awareness ganhou arquivo persistido, frequencia configuravel, backlog/tipos de falha/tendencia de surprise no diagnostico, fila deduplicada de diretivas e reaplicacao de diretivas `PROMPT` aprovadas no runtime.
- `introspect_self` deixou de responder por template cego e usa `self_model`, `latest_awareness` e `workspace`; a UI passou a exibir modelos/backends, routing policy, permissoes, `self_model_ref`, `capability_snapshot`, `pending_questions` e reduziu polling constante para tabs nao live.
- A hermeticidade foi fechada com `LocalBridgeStateStore` configuravel, paths derivados de runtime temporario, `sys.executable` na CLI, imports opcionais mais seguros e bateria verde em harness, unittest completo, `tsc --noEmit` e `vite build`.

## Progress

- [x] Frente 1 - Routing Policy Operacional e Capability Health Coerente
- [x] Frente 2 - Awareness Loop Real, Persistencia e Diretivas Auditaveis
- [x] Frente 3 - Introspecao Grounded e UI Completa
- [x] Frente 4 - Hermeticidade de CLI, API e Testes

## Decision Log

- 2026-03-29: Plano aberto para suceder `2026-03-29-ai-os-application-sprints.md` apos revisao de cobertura indicar lacunas entre superficie implementada e gates prometidos.
- 2026-03-29: Decidido separar encerramento documental do ciclo anterior de um novo plano ativo de fechamento, para evitar estado ambiguo entre `active/` e `completed/`.
- 2026-03-29: Decidido tratar routing policy, awareness persistido, introspecao grounded e hermeticidade de testes como frente unica de fechamento, e nao como debt difuso.
- 2026-03-29: Plano encerrado apos validacao com `PYTHONPATH=src python3 -m calosum.harness_checks`, `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t .`, `tsc --noEmit` e `vite build --outDir /tmp/calosum-ui-build`.
