# Calosum

Framework de agente IA neuro-simbólico com arquitetura cognitiva de duplo hemisfério. Combina percepção contínua baseada em embeddings, raciocínio simbólico via LLM, execução segura de ações tipadas e metacognição evolutiva inspirada em Group-Evolving Agents (GEA). Projetado para ser legível, verificável e progressivamente resiliente.

---

## Índice

- [Arquitetura](#arquitetura)
- [Pipeline cognitivo](#pipeline-cognitivo)
- [Conceitos principais](#conceitos-principais)
- [Estrutura de módulos](#estrutura-de-módulos)
- [Tipos e Contratos](#tipos-e-contratos)
- [Instalação](#instalação)
- [Uso rápido](#uso-rápido)
- [API HTTP](#api-http)
- [Perfis de infraestrutura](#perfis-de-infraestrutura)
- [Variáveis de ambiente](#variáveis-de-ambiente)
- [Docker](#docker)
- [UI de telemetria](#ui-de-telemetria)
- [Aprendizado contínuo](#aprendizado-contínuo)
- [Testes](#testes)
- [Desenvolvimento](#desenvolvimento)
- [Governança arquitetural](#governança-arquitetural)

---

## Arquitetura

```
UserTurn
  │
  ▼
RightHemisphere ────── percepção, emoção, salience, surprise
  │
  ▼
CognitiveTokenizer ─── bridge Information Bottleneck (soft prompts, corpus caloso)
  │
  ▼
LeftHemisphere ──────── raciocínio simbólico, programas lambda tipados, ações
  │
  ▼
StrictLambdaRuntime ─── validação e execução segura de ações
  │
  ▼
HeuristicVerifier ───── crítica (loop de reparo guiado)
  │
  ▼  [se surprise > 0.6 ou ambiguity > 0.8]
GEAReflectionController ─ group turns: variantes cognitivas competidoras + neuroplasticidade
  │
  ▼
IntrospectionEngine + EvolutionProposer ─ auto-diagnóstico e diretivas evolutivas
```

O projeto segue o padrão **Ports and Adapters** com fronteiras de dependência verificadas estaticamente via análise AST em `harness_checks.py`. Violações de camada quebram o build imediatamente.

### Camadas

| Camada | Responsabilidade | Regras |
|--------|-----------------|--------|
| `shared/` | Tipos, Protocols, utilitários puros | Zero dependências internas; sem SDKs externos |
| `domain/` | Lógica cognitiva central | **Nunca** importa de `adapters/` ou `bootstrap/`; sem torch, transformers, httpx |
| `adapters/` | Implementações concretas dos Protocols | Todo código ML (`torch`, `transformers`, `peft`) vive aqui exclusivamente |
| `bootstrap/` | Ponto de entrada e injeção de dependência | Único lugar que instancia adapters e os injeta no domain |

---

## Pipeline cognitivo

### Turno simples (surprise ≤ 0.6)

```
1. CalosumAgent.aprocess_turn(user_turn)
2. MemorySystem.abuild_context()          → MemoryContext
3. RightHemisphere.aperceive()            → RightHemisphereState (latent_vector, emotional_labels, surprise_score)
4. surprise ≤ 0.6 AND ambiguity ≤ 0.8?
5. CognitiveTokenizer.atranslate()        → CognitiveBridgePacket (soft_prompts, control_signal, directives)
6. LeftHemisphere.areason()              → LeftHemisphereResult (response_text, lambda_program, actions[])
7. StrictLambdaRuntime.arun()            → ActionExecutionResult[]
8. HeuristicVerifier.averify()           → CritiqueVerdict
9. Se inválido: AgentExecutionEngine retry com LeftHemisphere.arepair() + critique_feedback
10. Armazenar MemoryEpisode + registrar telemetria
11. A cada N turnos: _awareness_loop → IntrospectionEngine + EvolutionProposer
```

### Group Turn (surprise > 0.6 ou ambiguity > 0.8)

```
1. Reutilizar: memory_context, right_state, workspace do turno atual
2. Para cada variante cognitiva [analitico, empatico, pragmatico]:
   a. Clonar CognitiveTokenizer com tokenizer_overrides da variante
   b. Clonar LeftHemisphere com left_overrides da variante
   c. AgentExecutionEngine.run_variant() com directives da variante
   d. Coletar CognitiveCandidate(variante, turn_result)
3. GEAReflectionController.aevaluate(candidates, base_tokenizer)
   - Pontuar: empathy (0.5) + runtime safety (0.3) + action simplicity (0.2)
   - Selecionar vencedor por score mais alto
   - Propor bridge_adjustments para neuroplasticidade
4. apply_neuroplasticity(tokenizer, outcome)
   - Atualizar tokenizer.config via interpolação lerp(atual, winner.param, 0.05)
   - Persistir estado de adaptação
5. Armazenar episódio do vencedor + registrar reflection event na telemetria
6. Retornar GroupTurnResult com todos os candidatos, reflection e result selecionado
```

### Sleep Mode (consolidação e treino noturno)

```
1. MemorySystem.asleep_mode()
   - SleepModeConsolidator: promove emoções frequentes → regras semânticas
   - Extrai knowledge triples do episódico
   - Gera datasets de treino (ShareGPT e DSPy format)
   - Retorna ConsolidationReport
2. NightTrainer.arun_training_cycle()
   - DSPy: otimização de prompts via métricas do consolidation data
   - LoRA/QLoRA: fine-tuning leve dos pesos do modelo
   - Exporta prompts otimizados + checkpoints adaptados
```

### Idle Foraging (geração endógena de objetivos)

```
1. CalosumAgent.aidle_foraging()
2. Consulta knowledge graph por lacunas / conhecimento desatualizado
3. Gera UserTurn sintético com prompt epistêmico
4. Executa aprocess_turn() → agente usa search_web, read_file, execute_bash
5. Atualiza memória com novo conhecimento descoberto
```

---

## Conceitos principais

### Hemisfério Direito — Percepção & Emoção

O `RightHemisphereJEPA` (heurístico) e `HuggingFaceRightHemisphereAdapter` convertem texto de entrada em representações latentes contínuas. Ambos os backends implementam agora percepção **semanticamente ancorada** com calibração de salience por sessão, bias de feedback de runtime e provenance explícita de telemetria.

**Backends disponíveis** (degradação automática):
1. HuggingFace `sentence-transformers` — MiniLM-L6-v2 ou modelo configurado via `CALOSUM_EMBEDDING_MODEL`
2. JEPA heurístico — embedding determinístico baseado em seed hash (sem GPU, sem dependências opcionais)

**Extração afetiva** (`_extract_emotional_labels`):
- Passo 1: keyword matching exato nos `salience_keywords` configurados
- Passo 2: similaridade de cosseno com embeddings de labels emocionais usando **thresholds por label** (calibrados individualmente por emoção — ex: `urgente: 0.42`, `feliz: 0.50`, `desespero: 0.42`)
- Retorna `(labels, emotion_meta)` com `keyword_hits`, `vector_hits`, `peak_similarity`

**Calibração de salience** (`_calibrate_salience`):
```
raw_salience = _estimate_salience(text, labels)
runtime_bias = _runtime_feedback_bias(workspace)   # feedback do runtime anterior
salience_input = min(1.0, raw_salience + runtime_bias)

# Suavização EMA + janela deslizante + limitador de variação
moving_avg = mean(history[-salience_window_size:])
blended = alpha * salience_input + (1 - alpha) * moving_avg
calibrated = clamp(blended, previous ± salience_max_step)  # sem picos abruptos

# Parâmetros padrão: window=6, alpha=0.45 (HF) / 0.5 (JEPA), max_step=0.22
```

**Bias de feedback de runtime** (`_runtime_feedback_bias`):
```
# Lê workspace.task_frame["previous_runtime_feedback"] (últimas 3 tentativas)
rejection_rate = rejected / (rejected + executed)
intensity = min(0.15, rejection_rate * 0.12 + attempts * 0.01)
# Aumenta salience quando o runtime está tendo muitas rejeições
```

**Confiança dinâmica** (`_estimate_confidence`) — exclusivo HuggingFace:
```
base = 0.55
+ texto ≥ 20 chars  → +0.08
+ sinais multimodais → +0.04 por sinal (max 0.12)
+ keyword_hits       → +0.04 por hit (max 0.10)
+ vector_hits        → +0.03 por hit (max 0.08)
+ peak_similarity ≥ 0.6 → +0.08
- sem labels         → -0.06
resultado: clamp(0.35, 0.95)
```

**Saída** (`RightHemisphereState`):
- `latent_vector`: vetor 384-dim da entrada
- `salience`: 0.0–1.0 calibrado por sessão (sem picos falsos)
- `emotional_labels`: lista de rótulos (`["urgente", "técnico", "informativo", ...]`)
- `world_hypotheses`: dicionário com `urgencia`, `complexidade`, `diversidade_sensorial`, `operational_risk`
- `surprise_score`: energia livre estimada (ver Inferência Ativa)
- `confidence`: confiança dinâmica (0.35–0.95)

**Telemetria padronizada** (campos estáveis em ambos os backends):
```json
{
  "right_backend": "huggingface_sentence_transformers | heuristic_jepa",
  "right_model_name": "all-MiniLM-L6-v2",
  "right_mode": "embedding | heuristic",
  "degraded_reason": null,
  "raw_salience": 0.72,
  "runtime_feedback_bias": 0.03,
  "emotion_keyword_hits": 2,
  "emotion_vector_hits": 1,
  "emotion_peak_similarity": 0.551
}
```

### Workspace Cognitivo — Contexto Compartilhado por Turno

O `CognitiveWorkspace` é inicializado no início de cada turno via `init_turn_workspace()` e carrega contexto de continuidade da sessão anterior:

```python
previous_workspace = agent.last_workspace_by_session.get(session_id)
task_frame = {
    "session_id": ...,
    "turn_id": ...,
    "user_text": ...,
    "previous_runtime_feedback": previous_workspace.runtime_feedback[-3:],  # últimas 3 tentativas
    "previous_verifier_feedback": previous_workspace.verifier_feedback[-2:], # últimas 2 críticas
}
```

Esse mecanismo fecha o loop bidirecional: o hemisfério direito lê `previous_runtime_feedback` para ajustar o `runtime_feedback_bias` na percepção do turno atual — falhas de execução anteriores aumentam a salience percebida, priorizando cautela no ciclo seguinte.

### Bridge / Corpus Caloso — Information Bottleneck

O `CognitiveTokenizer` projeta o estado latente contínuo em tokens discretos simbólicos via camada PyTorch (384 → 64 → 7 neurônios):

```
latent_vector (384-dim)
  → Linear(384, 64) → ReLU()
  → Linear(64, 7) → Sigmoid()
  → [salience_neuron, weight_0..5]
```

**Saída** (`CognitiveBridgePacket`):
- `soft_prompts`: lista de `SoftPromptToken(token, weight, provenance="neural_bottleneck")`
- `control`: `BridgeControlSignal` com `target_temperature`, `empathy_priority`, `system_directives[]`
- `salience`: valor consolidado de salience

Os parâmetros (`base_temperature`, `salience_threshold`, `empathy_gain`) são ajustados a cada group turn via neuroplasticidade, persistidos em arquivo JSON entre sessões.

### Hemisfério Esquerdo — Raciocínio & Ações

O `LeftHemisphereLogicalSLM` (esqueleto de domain) delega para adapters LLM (`llm_failover.py`, `llm_qwen.py`) que geram:

**Saída** (`LeftHemisphereResult`):
- `response_text`: resposta em linguagem natural
- `lambda_program`: `TypedLambdaProgram` com `signature`, `expression`, `expected_effect`
- `actions[]`: lista de `PrimitiveAction(action_type, typed_signature, payload{}, safety_invariants[])`
- `reasoning_summary[]`: cadeia de raciocínio textual

**Ações registradas**:

| Ação | Descrição |
|------|-----------|
| `respond_text` | Retornar texto ao usuário |
| `propose_plan` | Gerar plano estruturado |
| `load_semantic_rules` | Consultar regras semânticas da memória |
| `search_web` | Busca DuckDuckGo |
| `write_file` | Escrever arquivo no workspace |
| `read_file` | Ler arquivo do workspace |
| `execute_bash` | Sandbox Python seguro |
| `introspect_self` | Auto-modelo e diagnóstico |
| `code_execution` | Execução de código isolada |
| `http_request` | Chamadas HTTP com vault de credenciais |

O hemisfério esquerdo recebe feedback de critique na tentativa de reparo (`repair()` / `arepair()`) e carrega prompts otimizados pelo DSPy night trainer.

### Inferência Ativa — Cálculo de Surpresa

```python
# Vetores recentes da memória
distances = [cosine_distance(current_latent, past_latent) for past in recent_episodes]

# Distribuição com peso de recência
prior = recency_weighted_distribution(len(distances))

# Decomposição da energia livre (ActiveInferenceConfig)
posterior  = softmax(-distance_temperature * distances)   # temperature=4.0
complexity = KL(posterior || prior)                       # distância do modelo base
ambiguity  = entropy(posterior)                           # incerteza nas hipóteses
novelty    = min(distances) / 2.0                         # quão novo é o contexto
free_energy = complexity + ambiguity + (novelty_weight * novelty)  # novelty_weight=0.25

# Normalização logarítmica pelo tamanho do histórico
scale      = max(1.0, 1.0 + log(len(distances) + 1))
surprise   = min(1.0, free_energy / scale)

# Telemetria: free_energy, free_energy_complexity, free_energy_ambiguity,
#             free_energy_novelty, posterior_peak, memory_alignment

# Despacho:
# surprise ≤ 0.6 AND ambiguity ≤ 0.8 → turno simples
# caso contrário → group turn com 3 variantes cognitivas
```

### Group Turns — GEA (Group-Evolving Agents)

| Variante | Temperature | Prioridade | Threshold salience |
|----------|-------------|------------|-------------------|
| `analitico` | 0.18 | Consistência lógica, verificabilidade | baixo |
| `empatico` | 0.34 | Contexto afetivo, linguagem segura | baixo |
| `pragmatico` | 0.22 | Ações mínimas, resposta concisa | padrão |

**Scoring do GEAReflectionController**:
```
empathy_score    = -len(critique_issues) / 100          (peso 0.5)
safety_score     = execuções_bem_sucedidas / total       (peso 0.3)
simplicity_score = 1.0 / (1.0 + action_count)           (peso 0.2)
total_score      = 0.5*empathy + 0.3*safety + 0.2*simplicity
```

### Sistema de Memória Dual

**Stores disponíveis** (degradação automática: Qdrant → JSONL → RAM):

| Store | Tipo | Backend |
|-------|------|---------|
| `InMemoryEpisodicStore` | Turnos completos rankeados por relevância | RAM |
| `InMemorySemanticStore` | Regras com score de força | RAM |
| `InMemorySemanticGraphStore` | Triplas de conhecimento (sujeito, predicado, objeto, peso) | RAM |
| `PersistentDualMemorySystem` | Todos os stores acima | JSONL em `.calosum-runtime/` |
| `QdrantDualMemoryAdapter` | Busca vetorial semântica | Qdrant + embeddings |

**Queries com escopo de sessão**: todos os backends (RAM, JSONL, Qdrant) filtram episódios pela `session_id` atual antes de retornar resultados — episódios de outras sessões só são incluídos como fallback quando não há histórico da sessão corrente.

O `MemoryContext` fornecido ao hemisfério esquerdo contém: `recent_episodes[]`, `semantic_rules[]`, `knowledge_triples[]`.

### Runtime Seguro — Lambda DSL

O `StrictLambdaRuntime` valida cada programa lambda antes da execução:

1. Verifica se o tipo de ação está no `ToolRegistry`
2. Alinha ações declaradas com o `lambda_program.expression`
3. Verifica presença de `safety_invariants` por ação
4. Rejeita imediatamente ações desconhecidas ou efeitos colaterais não autorizados

O `LambdaExecutionPlanner` parseia a expressão DSL → plano de execução com dependências entre ações. Falhas são classificadas em: `SCHEMA_VIOLATION`, `UNSAFE_CONTENT`, `RUNTIME_REJECTION`, `INCOMPLETE_RESULT`.

### Loop de Reparo (AgentExecutionEngine)

O `AgentExecutionEngine` coordena retentativas com feedback orientado:

```
Tentativa 1: LeftHemisphere.areason() → runtime → verifier
Se inválido:
  Tentativa 2..N: LeftHemisphere.arepair(
    previous_result=resultado_anterior,
    rejected_results=histórico_rejeições,
    critique_feedback=críticas_do_verifier
  )
```

Cada tentativa de reparo recebe o `CritiqueVerdict` completo com `critique_reasoning[]`, `identified_issues[]` e `suggested_fixes[]`.

Após cada tentativa, o engine persiste no workspace o histórico de execução:
```python
workspace.runtime_feedback.append({
    "attempt": retry_count,
    "executed_count": len(executed_results),
    "rejected_count": len(rejected_results),
    "tool_success_rate": executed / total,
    "critique_valid": critique_verdict.is_valid,
})
```

Esse histórico é carregado no `task_frame` do turno **seguinte** da sessão e influencia o `runtime_feedback_bias` do hemisfério direito — fechando o loop bidirecional percepção ↔ execução.

Desvios cognitivos detectados no `reasoning_summary` (`mismatch`, `override`, `false alarm`) são registrados em `workspace.left_notes["cognitive_override_detected"]` e propagados na telemetria.

### Evolução e Introspecção

O **awareness loop** é executado a cada N turnos (configurável):

```
1. IntrospectionEngine.analyze(dashboard_telemetria) → SessionDiagnostic
   - tool_success_rate, average_retries, average_surprise
   - bottlenecks[], failure_types{}, surprise_trend
   - dominant_variant, dominant_variant_ratio

2. EvolutionProposer.propose(diagnostic) → EvolutionDirective[]
   - tool_success < 70%        → aumentar retries (PARAMETER) → auto-aplicado
   - backlog de aprovações > 0 → pedir esclarecimento (PROMPT) → fila manual
   - colapso de variante       → diversificar (PROMPT) → fila manual
   - surprise crescente        → reduzir comprometimento (PROMPT) → fila manual

3. JsonlEvolutionArchive: persiste diretivas pending/applied entre sessões
```

O `build_self_model()` gera `CognitiveArchitectureMap` com componentes, conexões e `adaptation_surface` (parâmetros tunable), exposto via `/v1/system/architecture`.

---

## Estrutura de módulos

```
src/calosum/
│
├── shared/                         # Tipos e contratos (sem dependências internas)
│   ├── types.py                    # Dataclasses centrais (UserTurn, RightHemisphereState, ...)
│   ├── ports.py                    # Protocols injetáveis (9 contratos)
│   ├── schemas.py                  # Schemas Pydantic para outputs estruturados
│   ├── tools.py                    # ToolRegistry, ToolSchema, ToolDescriptor
│   ├── async_utils.py              # maybe_await, run_sync
│   └── serialization.py            # Conversão JSON / primitivos
│
├── domain/                         # Lógica cognitiva pura (sem SDKs externos)
│   ├── orchestrator.py             # CalosumAgent — pipeline principal
│   ├── bridge.py                   # CognitiveTokenizer (Information Bottleneck)
│   ├── right_hemisphere.py         # RightHemisphereJEPA (heurístico)
│   ├── left_hemisphere.py          # LeftHemisphereLogicalSLM
│   ├── memory.py                   # DualMemorySystem (episódica + semântica)
│   ├── persistent_memory.py        # Camada JSONL / Qdrant
│   ├── runtime.py                  # StrictLambdaRuntime + execução segura
│   ├── runtime_dsl.py              # LambdaExecutionPlanner + validação DSL
│   ├── metacognition.py            # GEAReflectionController + personas cognitivas
│   ├── verifier.py                 # HeuristicVerifier (critique / CRITIC-like)
│   ├── telemetry.py                # CognitiveTelemetryBus + sinks
│   ├── event_bus.py                # InternalEventBus (eventos cognitivos internos)
│   ├── agent_execution.py          # AgentExecutionEngine (retry / repair)
│   ├── workspace.py                # CognitiveWorkspace (contexto compartilhado do turno)
│   ├── self_model.py               # build_self_model → CognitiveArchitectureMap
│   ├── evolution.py                # EvolutionProposer + JsonlEvolutionArchive
│   ├── introspection.py            # IntrospectionEngine → SessionDiagnostic
│   └── multiagent.py               # Raciocínio multi-papel (Planner / Executor / Verifier)
│
├── adapters/                       # Implementações concretas (ML, SDKs externos)
│   ├── right_hemisphere_hf.py      # HuggingFace sentence-transformers (MiniLM-L6-v2)
│   ├── active_inference.py         # Surpresa via energia livre (pymdp ou numpy)
│   ├── llm_failover.py             # Roteamento multi-provedor com cooldown
│   ├── llm_qwen.py                 # Integração Qwen
│   ├── llm_payloads.py             # Formatação de output estruturado (Responses API)
│   ├── text_embeddings.py          # OpenAI / HuggingFace / lexical (cadeia de fallback)
│   ├── memory_qdrant.py            # QdrantDualMemoryAdapter (busca vetorial)
│   ├── action_runtime.py           # ConcreteActionRuntime → ToolRegistry
│   ├── bridge_store.py             # Persistência do estado de adaptação do bridge
│   ├── telemetry_otlp.py           # Exportador OTLP → Jaeger (W3C traces)
│   ├── knowledge_graph_nanorag.py  # NanoGraphRAG (opcional, fallback NetworkX)
│   ├── channel_telegram.py         # Canal bidirecional Telegram
│   ├── night_trainer.py            # Base de aprendizado contínuo (orquestrador)
│   ├── night_trainer_dspy.py       # Otimização de prompts via DSPy
│   ├── night_trainer_lora.py       # Fine-tuning QLoRA noturno
│   └── tools/
│       ├── code_execution.py       # Sandbox Python seguro
│       └── http_request.py         # Wrapper HTTP com vault de credenciais
│
├── bootstrap/                      # Injeção de dependência e entrypoints
│   ├── factory.py                  # CalosumAgentBuilder — orquestra todas as dependências
│   ├── settings.py                 # InfrastructureSettings — resolução de env vars
│   ├── api.py                      # Servidor FastAPI (SSE, webhooks, telemetria)
│   └── cli.py                      # CLI: chat, run-turn, run-scenario, sleep, idle
│
└── harness_checks.py               # Governança AST: importações, tamanho de módulo, planos
```

---

## Tipos e Contratos

### Tipos centrais (`shared/types.py`)

```python
UserTurn(session_id, user_text, signals[], turn_id)

RightHemisphereState(
    context_id, latent_vector[], salience, emotional_labels[],
    world_hypotheses{}, confidence, surprise_score
)

CognitiveBridgePacket(
    context_id, soft_prompts[], control, salience, bridge_metadata
)

BridgeControlSignal(
    target_temperature, empathy_priority, system_directives[]
)

SoftPromptToken(token, weight, provenance)

LeftHemisphereResult(
    response_text, lambda_program, actions[], reasoning_summary[]
)

TypedLambdaProgram(signature, expression, expected_effect)

PrimitiveAction(
    action_type, typed_signature, payload{}, safety_invariants[]
)

ActionExecutionResult(
    action_type, typed_signature, status, output{}, violations[]
)

CritiqueVerdict(
    is_valid, critique_reasoning[], identified_issues[],
    suggested_fixes[], confidence, failure_types[]
)

MemoryEpisode(
    episode_id, recorded_at, user_turn, right_state, bridge_packet,
    left_result, execution_results[], runtime_retry_count, critique_revision_count
)

MemoryContext(recent_episodes[], semantic_rules[], knowledge_triples[])

CognitiveWorkspace(
    task_frame{}, self_model_ref, capability_snapshot,
    right_notes{}, bridge_state{}, left_notes{},
    verifier_feedback[], runtime_feedback[], pending_questions[]
)

SessionDiagnostic(
    session_id, analyzed_turns, tool_success_rate, average_retries,
    average_surprise, bottlenecks[], failure_types{},
    pending_approval_backlog, surprise_trend,
    dominant_variant, dominant_variant_ratio
)

EvolutionDirective(
    directive_id, directive_type, target_component,
    proposed_change{}, reasoning, status
)

CognitiveArchitectureMap(components[], connections[], adaptation_surface, capabilities)
```

### Protocols injetáveis (`shared/ports.py`)

```python
RightHemispherePort
  .perceive(user_turn, memory_context, workspace) → RightHemisphereState
  .aperceive(...)                                 → async RightHemisphereState

CognitiveTokenizerPort
  .translate(right_state, workspace)  → CognitiveBridgePacket
  .atranslate(...)                    → async CognitiveBridgePacket

LeftHemispherePort
  .reason(user_turn, bridge_packet, memory_context, runtime_feedback, attempt, workspace)
    → LeftHemisphereResult
  .repair(user_turn, bridge_packet, memory_context, previous_result,
          rejected_results, attempt, critique_feedback, workspace)
    → LeftHemisphereResult

MemorySystemPort
  .build_context(user_turn, episodic_limit) → MemoryContext
  .store_episode(episode)                   → None
  .sleep_mode()                             → ConsolidationReport

ActionRuntimePort
  .run(left_result, workspace)   → list[ActionExecutionResult]
  .get_registered_tools()        → list[ToolDescriptor]

VerifierPort
  .verify(user_turn, left_result, execution_results, workspace) → CritiqueVerdict

TelemetryBusPort
  .record_turn(result)                    → None
  .record_reflection(session_id, turn_id, payload) → None
  .dashboard_for_session(session_id)      → dict[str, list[dict]]

ReflectionControllerPort
  .evaluate(candidates[], base_tokenizer) → ReflectionOutcome
  .apply_neuroplasticity(tokenizer, outcome) → None
```

---

## Instalação

**Requisitos**: Python 3.11+, Node.js 18+ (somente para a UI)

```bash
# Clonar e instalar dependências
git clone <repo>
cd calosum
pip install -r requirements.txt

# Verificar integridade arquitetural antes de qualquer mudança
PYTHONPATH=src python3 -m calosum.harness_checks
```

**Dependências principais**:

| Pacote | Uso |
|--------|-----|
| `pydantic >= 2.5` | Validação de schemas |
| `fastapi >= 0.100` + `uvicorn` | Servidor HTTP / SSE |
| `httpx >= 0.25` | Cliente HTTP assíncrono |
| `qdrant-client >= 1.7` | Banco de vetores |
| `sentence-transformers >= 3.0` | Embeddings semânticos |
| `transformers >= 4.40` | Modelos HuggingFace |
| `torch >= 2.1` | Bridge neural + LoRA |
| `tenacity >= 8.3` | Retry com backoff |
| `dspy` | Otimização de prompts |
| `peft >= 0.18` | Fine-tuning LoRA/QLoRA |
| `nano-graphrag` | Knowledge graph (opcional) |
| `inferactively-pymdp` | Inferência ativa (opcional) |
| `python-telegram-bot >= 21` | Canal Telegram (opcional) |
| `sse-starlette >= 1.8` | Server-Sent Events |
| `duckduckgo-search >= 5.0` | Ferramenta de busca web |

---

## Uso rápido

```bash
# Chat interativo (REPL)
python3 -m calosum.bootstrap.cli chat

# Turno único
python3 -m calosum.bootstrap.cli run-turn \
  --session-id demo \
  --text "sua mensagem" \
  --infra-profile persistent

# Servidor HTTP
python3 -m calosum.bootstrap.api
# POST http://localhost:8000/v1/chat/completions

# UI de telemetria
cd ui && npm run dev   # http://localhost:5173

# Consolidação de memória + treino noturno
python3 -m calosum.bootstrap.cli sleep

# Foraging endógeno (geração de objetivos idle)
python3 -m calosum.bootstrap.cli idle
```

---

## API HTTP

O servidor FastAPI serializa turnos por sessão via **per-session concurrency lanes** (`asyncio.Lock` por `session_id`) — múltiplos clientes na mesma sessão nunca processam turnos em paralelo, garantindo consistência de workspace e memória.

O servidor expõe os seguintes endpoints:

### Inferência

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `POST` | `/v1/chat/completions` | Turno cognitivo completo (streaming SSE) |

**Body** (`application/json`):
```json
{
  "session_id": "demo",
  "messages": [{"role": "user", "content": "sua mensagem"}],
  "stream": true
}
```

**Resposta SSE**: eventos `data: {...}` com campos `delta`, `turn_id`, `surprise_score`, `variant_selected`.

### Sistema

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/v1/system/info` | Versão, perfil ativo, capacidades |
| `GET` | `/v1/system/architecture` | `CognitiveArchitectureMap` completo |
| `GET` | `/v1/system/capabilities` | Ferramentas registradas e status |
| `GET` | `/v1/system/state` | Estado atual do agente (bridge params, variante dominante) |

### Telemetria

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/v1/telemetry/dashboard/{session_id}` | Timeline de eventos por canal cognitivo |

**Canais de telemetria retornados**: `felt` (percepção), `thought` (raciocínio), `decision` (ações), `execution` (runtime), `reflection` (GEA), `sleep` (consolidação).

---

## Perfis de infraestrutura

Controlados pela variável `CALOSUM_INFRA_PROFILE`:

| Perfil | Comportamento | Uso |
|--------|---------------|-----|
| `ephemeral` (padrão) | RAM-only. Sem persistência. | Testes unitários, prototipagem rápida |
| `persistent` | JSONL em `.calosum-runtime/`. | Desenvolvimento local, single-machine |
| `docker` | Qdrant + OTLP + Jaeger. | Produção, multi-machine, observabilidade |

**Cadeia de degradação** (o bootstrap nunca falha por infraestrutura ausente):

```
Memória:          Qdrant      → JSONL         → RAM
Hemisfério D.:    HuggingFace → JEPA heurístico
Embeddings:       OpenAI API  → OpenAI compat. → HuggingFace → lexical determinístico
Knowledge Graph:  NanoGraphRAG → NetworkX fallback
LLM:              endpoint principal → endpoint fallback (com cooldown automático)
```

---

## Variáveis de ambiente

### LLM principal

| Variável | Descrição |
|----------|-----------|
| `CALOSUM_LEFT_ENDPOINT` | Endpoint da API (ex: `https://api.openai.com/v1`) |
| `CALOSUM_LEFT_API_KEY` | Chave de API |
| `CALOSUM_LEFT_MODEL` | Modelo (ex: `gpt-4o`, `qwen2.5-72b`) |
| `CALOSUM_LEFT_PROVIDER` | `openai` ou `qwen` (autodetectado pelo endpoint) |
| `CALOSUM_LEFT_REASONING_EFFORT` | `low` / `medium` / `high` (extended thinking) |

### LLM de fallback

| Variável | Descrição |
|----------|-----------|
| `CALOSUM_LEFT_FALLBACK_ENDPOINT` | Endpoint alternativo |
| `CALOSUM_LEFT_FALLBACK_API_KEY` | Chave alternativa |
| `CALOSUM_LEFT_FALLBACK_MODEL` | Modelo alternativo |
| `CALOSUM_LEFT_FALLBACK_PROVIDER` | Provider do fallback |
| `CALOSUM_LEFT_FALLBACK_REASONING_EFFORT` | Esforço de raciocínio do fallback |

### Roteamento por papel cognitivo

| Variável | Papel |
|----------|-------|
| `CALOSUM_PERCEPTION_MODEL` | Hemisfério direito (percepção) |
| `CALOSUM_REASON_MODEL` | Hemisfério esquerdo (raciocínio) |
| `CALOSUM_REFLECTION_MODEL` | GEA reflection scoring |
| `CALOSUM_VERIFIER_MODEL` | Verificador (critique) |

### Backends 2026 (dual-hemisphere)

| Variável | Descrição |
|----------|-----------|
| `CALOSUM_RIGHT_BACKEND` | `auto` / `vjepa21` / `vljepa` / `jepars` / `huggingface` |
| `CALOSUM_RIGHT_MODEL_PATH` | Diretório local com artefatos do hemisfério direito |
| `CALOSUM_RIGHT_ACTION_CONDITIONED` | `true`/`false` para predição condicionada por ação |
| `CALOSUM_RIGHT_HORIZON` | Horizonte preditivo do world model |
| `CALOSUM_RIGHT_JEPARS_BINARY` | Binário local do backend Rust `jepa-rs` |
| `CALOSUM_LEFT_BACKEND` | `rlm` / `qwen` (default legado) |
| `CALOSUM_LEFT_RLM_RUNTIME_COMMAND` | Comando do runtime RLM local |
| `CALOSUM_LEFT_RLM_PATH` | Caminho local do modelo/artefato RLM |
| `CALOSUM_LEFT_RLM_MAX_DEPTH` | Profundidade máxima de recursão do RLM |
| `CALOSUM_BRIDGE_BACKEND` | `heuristic` / `cross_attention` |
| `CALOSUM_GEA_SHARING_ENABLED` | Habilita experience sharing persistente no GEA |
| `CALOSUM_GEA_EXPERIENCE_STORE_PATH` | Caminho do SQLite de experiência do GEA |

### Infraestrutura

| Variável | Descrição |
|----------|-----------|
| `CALOSUM_INFRA_PROFILE` | `ephemeral` / `persistent` / `docker` |
| `CALOSUM_MEMORY_DIR` | Diretório de memória JSONL (padrão: `.calosum-runtime/`) |
| `CALOSUM_VECTORDB_URL` | Qdrant (ex: `http://localhost:6333`) |
| `CALOSUM_EMBEDDING_ENDPOINT` | Serviço de embeddings customizado |
| `CALOSUM_EMBEDDING_API_KEY` | Chave do serviço de embeddings |
| `CALOSUM_EMBEDDING_MODEL` | Modelo de embedding (ex: `text-embedding-3-small`) |
| `CALOSUM_EMBEDDING_PROVIDER` | `openai` / `openai_compatible` / `huggingface` / `lexical` |
| `CALOSUM_OTEL_COLLECTOR_ENDPOINT` | Coletor OTLP (ex: `http://localhost:4318`) |
| `CALOSUM_JAEGER_UI_URL` | Interface Jaeger (ex: `http://localhost:16686`) |
| `CALOSUM_API_PORT` | Porta do servidor FastAPI (padrão: `8000`) |

### Vault de credenciais (ferramentas)

Variáveis prefixadas com `CALOSUM_VAULT_` são injetadas automaticamente no `ConcreteActionRuntime` como credenciais para as ferramentas HTTP.

```bash
CALOSUM_VAULT_GITHUB_TOKEN=ghp_...
CALOSUM_VAULT_SLACK_WEBHOOK=https://hooks.slack.com/...
```

### Telegram (opcional)

| Variável | Descrição |
|----------|-----------|
| `TELEGRAM_BOT_TOKEN` | Token do bot Telegram |
| `CALOSUM_TELEGRAM_DM_POLICY` | `open` (qualquer usuário, padrão) ou `allowlist` (apenas IDs permitidos) |
| `CALOSUM_TELEGRAM_ALLOWLIST` | IDs de usuários separados por vírgula (ex: `123456789,987654321`) |

Com `dm_policy=allowlist`, mensagens de remetentes não listados são descartadas silenciosamente com log de warning.

---

## Docker

```bash
# Subir toda a stack (API + Qdrant + OTLP + Jaeger)
docker compose -f deploy/docker-compose.yml up --build -d

# Logs
docker compose -f deploy/docker-compose.yml logs -f orchestrator
```

| Serviço | Porta | Descrição |
|---------|-------|-----------|
| `orchestrator` | 8000 | API FastAPI do Calosum |
| `qdrant` | 6333 | Banco de vetores (interface HTTP) |
| `otel-collector` | 4317 / 4318 | Coletor OpenTelemetry (gRPC / HTTP) |
| `jaeger` | 16686 | UI de rastreamento distribuído |

O `otel-collector-config.yaml` configura o pipeline OTLP: recebimento HTTP/gRPC → processamento em batch → exportação para Jaeger.

Modo local com JEPA no compose:

- `CALOSUM_MODE=local`
- `CALOSUM_JEPA_MODEL_PATH=/app/jepa_model`
- volume read-only: `./adapters/jepa_predictor:/app/jepa_model:ro`

---

## UI de telemetria

Interface React para visualização em tempo real do ciclo cognitivo.

```bash
cd ui && npm run dev     # desenvolvimento: http://localhost:5173
npm run build            # produção
```

**Funcionalidades**:
- Timeline de eventos cognitivos por canal (`felt`, `thought`, `decision`, `execution`, `reflection`)
- Scoreboard de variantes GEA (analitico / empatico / pragmatico)
- Gráfico de surpresa por turno e tendência de awareness
- Módulo de Geração Endógena de Objetivos (Idle Foraging)
- Inspeção de episódios de memória e regras semânticas consolidadas
- Visualização do auto-modelo (`CognitiveArchitectureMap`)

**Stack**: React 19, TypeScript, Vite, Tailwind CSS 4, lucide-react.

---

## Aprendizado contínuo

### Sleep Mode

```bash
python3 -m calosum.bootstrap.cli sleep
```

O `night_trainer` é agora injetado no `CalosumAgent` via `CalosumAgentBuilder.build()`. O `asleep_mode()` do orchestrator executa consolidação + treino em sequência e publica `SleepModeCompletedEvent` no event bus.

O `SleepModeConsolidator` analisa os episódios recentes e:

1. **Promoção semântica**: emoções e padrões frequentes viram regras semânticas com score de força
2. **Extração de triplas**: popula o knowledge graph a partir de conteúdo episódico
3. **Geração de datasets**: formatos ShareGPT (para LoRA) e DSPy (para otimização de prompts)

### Night Trainer

Três backends disponíveis (ativados automaticamente conforme disponibilidade):

| Backend | Arquivo | O que faz |
|---------|---------|-----------|
| **DSPy** | `night_trainer_dspy.py` | Otimiza prompts do hemisfério esquerdo via métricas de qualidade |
| **LoRA/QLoRA** | `night_trainer_lora.py` | Fine-tuning leve com `peft` nos dados consolidados |
| **OPRO-lite** | `night_trainer.py` | Otimização in-place de prompts sem dependências externas |

Os prompts otimizados são carregados automaticamente pelo `LeftHemisphereLogicalSLM` na próxima sessão.

---

## Testes

```bash
# Suite completa
PYTHONPATH=src python3 -m unittest discover -s tests -t .

# Arquivo específico
PYTHONPATH=src python3 -m unittest tests.test_pipeline

# Método específico
PYTHONPATH=src python3 -m unittest tests.test_runtime.TestStrictLambdaRuntime.test_reject_unknown_action

# Verificação de governança arquitetural
PYTHONPATH=src python3 -m calosum.harness_checks

# Benchmark local do hemisfério direito (gera JSON em docs/reports/)
PYTHONPATH=src .venv/bin/python examples/right_hemisphere_benchmark.py \
  --output-json docs/reports/2026-03-30-right-hemisphere-benchmark.json
```

### CI/CD com gates de qualidade

Pipeline em `.github/workflows/ci.yml`:

1. `Lint + Types`: `mypy --strict`, `ruff`, `harness_checks`.
2. `Unit Tests`: suíte unitária + gate de cobertura >= 80% em módulos novos/alterados.
3. `Integration`: pipeline com LLM mockado; falha se `latency_p95_ms > 5000`.
4. `Benchmark Gate`: comparação automática contra baseline; falha se regressão > 5% em `tool_success_rate`.

Os resultados automáticos de benchmark de CI são gerados em `docs/benchmarks/ci/` e publicados como artefatos de cada run.

**23 arquivos de teste** cobrindo todos os subsistemas principais:

| Arquivo | Cobertura |
|---------|-----------|
| `test_pipeline.py` | Integração completa do pipeline cognitivo |
| `test_runtime.py` | Validação do StrictLambdaRuntime |
| `test_memory.py` | Stores episódico, semântico e de grafos |
| `test_qdrant_adapter.py` | Adapter Qdrant com busca vetorial |
| `test_reflection.py` | GEAReflectionController e group turns |
| `test_factory.py` | CalosumAgentBuilder e injeção de dependência |
| `test_api.py` | Endpoints FastAPI e SSE |
| `test_llm_adapter.py` | Adapters LLM (OpenAI, Qwen) |
| `test_llm_failover.py` | Roteamento multi-provedor com cooldown |
| `test_night_trainer.py` | DSPy, LoRA e OPRO-lite |
| `test_awareness.py` | IntrospectionEngine e EvolutionProposer |
| `test_tool_registry.py` | ToolRegistry e validação de ações |
| `test_active_inference.py` | Cálculo de surpresa e energia livre |
| `test_telegram_channel.py` | Canal Telegram bidirecional |
| `test_self_model.py` | CognitiveArchitectureMap e auto-modelo |
| `test_telemetry_otlp.py` | Exportação OTLP para Jaeger |
| `test_knowledge_graph.py` | NanoGraphRAG e NetworkX fallback |
| `test_right_hemisphere_hf.py` | HuggingFace sentence-transformers |

---

## Fase atual do projeto (Mar/2026)

O projeto está na **Fase de Evolução do Hemisfério Direito** — tornando a percepção semanticamente ancorada, calibrada por memória e com provenance explícita, sem quebrar o modelo local-first.

### Status por fase

| Fase | Status |
|------|--------|
| Fundação arquitetural (Ports & Adapters, harness, pipeline) | Concluída |
| Runtime seguro e loop CRITIC-like | Concluída |
| Self-awareness (self-model, introspection, evolution, workspace) | Concluída |
| **Evolução do Hemisfério Direito (realidade perceptiva)** | **Em andamento** |
| Benchmark cognitivo local e gates de qualidade | Em execução contínua (CI gateado) |
| Pesquisa multimodal pesada (V-JEPA ou equivalente) | Backlog de pesquisa |

### Plano ativo: Right Hemisphere Reality Upgrade

Sprints do plano `docs/exec-plans/active/2026-03-30-right-hemisphere-reality-upgrade.md`:

| Sprint | Entregável | Status |
|--------|-----------|--------|
| Sprint 0 | Telemetria padronizada: `right_backend`, `right_mode`, `degraded_reason` em todos os adapters | **Concluída** |
| Sprint 1 | Extração afetiva calibrada: thresholds por label, confidence dinâmica, emotion_meta | **Concluída** |
| Sprint 2 | Surprise + salience calibrados por memória (salience smoothing, novelty, feedback bias) | **Concluída** |
| Sprint 3 | Fechamento de loop bidirecional: workspace carry-over entre turnos | **Concluída** |
| Sprint 4 | Benchmark local com comparativo heurístico vs embedding | Próxima |
| Sprint 5 | Adaptação contínua controlada (micro-ajustes de thresholds por reflexão) | Planejado |

### Princípios da fase atual

1. **Local-first pragmática**: sem dependência obrigatória de cloud
2. **Capacidades antes de hype**: medir ganho real antes de trocar stack
3. **Ports and Adapters**: toda evolução pesada atrás de interface
4. **Segurança e auditabilidade**: mudanças adaptativas pequenas, rastreáveis e reversíveis

### Horizontes de roadmap

| Horizonte | Período | Foco |
|-----------|---------|------|
| **H1** | 0–6 semanas | Hemisfério Direito realista, loop bidirecional, benchmark local |
| **H2** | 6–12 semanas | Adaptação contínua controlada, robustez do verifier, testes de regressão cognitiva |
| **H3** | 12+ semanas | Protótipo multimodal real (V-JEPA/M3-JEPA) atrás de adapter; estudos A/B |

---

## Desenvolvimento

### Adicionando uma nova integração

1. Defina um `Protocol` em `shared/ports.py` com os métodos sync e async.
2. Implemente o adapter em `adapters/` atrás desse Protocol (toda lib externa fica aqui).
3. Registre no `CalosumAgentBuilder` em `bootstrap/factory.py` com fallback adequado.
4. O `domain/` não deve saber da existência do adapter — use apenas o Protocol.
5. Se a mudança tocar mais de um subsistema, crie um plano em `docs/exec-plans/active/`.

### Adicionando uma nova ferramenta (ação)

1. Implemente a função em `adapters/tools/` com assinatura tipada.
2. Registre no `ToolRegistry` via `ConcreteActionRuntime` com `ToolDescriptor` completo.
3. Adicione o `action_type` à lista de ações executáveis do `StrictLambdaRuntime`.
4. O hemisfério esquerdo descobrirá a ferramenta via `MemoryContext.capabilities`.

### Executando um turno cognitivo local

```bash
# Com perfil persistent (persiste memória entre execuções)
CALOSUM_INFRA_PROFILE=persistent \
CALOSUM_LEFT_ENDPOINT=https://api.openai.com/v1 \
CALOSUM_LEFT_API_KEY=sk-... \
CALOSUM_LEFT_MODEL=gpt-4o \
python3 -m calosum.bootstrap.cli run-turn \
  --session-id minha-sessao \
  --text "explique sua arquitetura"
```

---

## Governança arquitetural

O `harness_checks.py` executa verificações AST automaticamente e falha o build em caso de violação:

| Verificação | Regra |
|-------------|-------|
| **Fronteiras de importação** | `domain/` nunca importa de `adapters/` ou `bootstrap/` |
| **SDKs externos** | `torch`, `transformers`, `httpx`, `openai` proibidos em `domain/` e `shared/` |
| **Tamanho de módulo** | Máximo 400 linhas por arquivo |
| **Planos de execução** | Mudanças em > 1 subsistema exigem plano em `docs/exec-plans/active/` |

```bash
# Executar antes de qualquer mudança estrutural
PYTHONPATH=src python3 -m calosum.harness_checks
```

**Convenções de documentação**:
- Português: semântica de domínio (rótulos emocionais, keywords de salience, passos de plano)
- Inglês: contratos técnicos, type hints, docstrings de API

**Dívida técnica**: registrada em `docs/exec-plans/tech-debt-tracker.md`.

**Planos completados**: 38 planos em `docs/exec-plans/completed/` documentam a evolução sistemática do projeto.
