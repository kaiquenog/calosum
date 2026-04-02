# Reliability

## Objetivos

- Todo turno deve produzir telemetria e latência observável.
- Rejeições do runtime devem acionar reparo automático limitado.
- O repositório deve falhar cedo quando arquitetura ou documentação derivarem.

## Controles Atuais

### Runtime e Repair Loop
- `CalosumAgentConfig.max_runtime_retries` (default 2): limite de tentativas de reparo por turno
- `AgentExecutionEngine._execute_with_retries()`: loop que chama `arun()` → `averify()` → `_repair_left_result()` até esgotar retries ou obter resultado válido
- `StrictLambdaRuntime` com rejeição explícita de ações não registradas
- `VerifierPort` / `CritiqueVerdict`: valida resultado final; feedback de critique_reasoning alimenta o repair

### Fallback por Camada
- Left hemisphere: `LLMFailoverAdapter` tenta endpoint primário e cai para secundário sem propagar erro
- Right hemisphere: HuggingFace com fallback para `ActiveInferenceRightHemisphereAdapter` heurístico
- Vector DB: Qdrant → JSONL → in-memory (`DualMemorySystem`)
- Embeddings: OpenAI → HuggingFace → léxico determinístico
- Knowledge graph: nano-graphrag → NetworkX in-memory

### Observabilidade
- `CognitiveTelemetryBus` com `trace_id`, `span_id` e métricas por turno
- `CognitiveTelemetrySnapshot` captura `felt`, `thought`, `decision`, `capabilities` (backend/health por componente) e `bridge_config` por turno
- `capability_snapshot` construído pelo builder reflete estado real de saúde dos backends (`ComponentHealth`: HEALTHY / DEGRADED / UNAVAILABLE)

### Governança do Repositório
- Harness checks validam: artefatos obrigatórios, links em AGENTS.md, refs em docs/index.md, headings de planos, tamanho de módulos, fronteiras de importação via AST

## SLOs Iniciais

- `process_turn`: telemetria obrigatória por turno; snapshot deve incluir capabilities
- `runtime_retry_count`: observável por decisão via `AgentTurnResult.retry_count`
- `group_turn`: resultado selecionado deve registrar `variant_id`, `selected_by` (`learned_model|rule_based|legacy`) e reflection event
- `sleep_mode`: não pode perder regras ou triplas já consolidadas

## Operação

- Rodar harness checks antes de mudanças estruturais: `PYTHONPATH=src python3 -m calosum.harness_checks`
- Rodar testes antes de merge: `PYTHONPATH=src python3 -m unittest discover -s tests -t .`
- Registrar debt estrutural no tracker: `docs/exec-plans/tech-debt-tracker.md`
