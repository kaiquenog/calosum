# Calosum

Framework de agente IA neuro-simbólico com arquitetura cognitiva de duplo hemisfério. Combina percepção baseada em embeddings, raciocínio via LLM, execução segura de ações e metacognição evolutiva inspirada em Group-Evolving Agents (GEA).

---

## Arquitetura

```
UserTurn
  │
  ▼
RightHemisphere ── percepção, emoção, salience
  │
  ▼
CognitiveTokenizer ── bridge (soft prompts, corpus caloso)
  │
  ▼
LeftHemisphere ── raciocínio, geração de ações
  │
  ▼
StrictLambdaRuntime ── execução segura
  │
  ▼
Verifier ── crítica e loop de reparo
  │
  ▼ [se surprise > 0.6 ou ambiguity > 0.8]
GEAReflectionController ── group turns (variantes cognitivas competidoras)
```

O projeto segue o padrão **Ports and Adapters** com fronteiras de dependência verificadas estaticamente por AST em `harness_checks.py`. Violações de camada quebram o build.

### Camadas

| Camada | Regra |
|--------|-------|
| `shared/` | Tipos, Protocols, utilidades puras. Zero dependências internas. |
| `domain/` | Lógica cognitiva central. **Nunca importa de `adapters/` ou `bootstrap/`**. Sem SDKs externos. |
| `adapters/` | Implementações concretas dos Protocols. Todo código ML (`torch`, `transformers`, `peft`) vive aqui. |
| `bootstrap/` | Único ponto de instanciação. Injeta adapters no domain via Factory. |

---

## Conceitos principais

### Duplo hemisfério

- **Hemisfério direito** (`RightHemisphereJEPA` / `HuggingFaceRightHemisphere`): processa percepção, estima salience (0–1), extrai labels emocionais e gera hipóteses de mundo (urgência, complexidade, diversidade sensorial).
- **Bridge / Corpus Caloso** (`CognitiveTokenizer`): projeta o estado latente em soft prompts discretos via Information Bottleneck. Os parâmetros (temperature, salience_threshold, directives) são ajustados por neuroplasticidade a cada turno de reflexão.
- **Hemisfério esquerdo** (`LeftHemisphereLogicalSLM`): recebe o bridge packet, consulta memória dual, gera programas lambda tipados com primitivas de ação seguras.

### Inferência ativa

O componente `ActiveInferenceMetrics` decompõe a surpresa em:

- **Complexity**: divergência do modelo base (distância latente)
- **Ambiguity**: incerteza nas hipóteses de mundo

A surpresa combinada (energia livre estimada) determina se o turno é processado diretamente ou escalado para um group turn.

### Group Turns e GEA

Quando `surprise > 0.6` ou `ambiguity > 0.8`, o orquestrador gera múltiplas variantes cognitivas em paralelo:

| Variante | Perfil |
|----------|--------|
| `analitico` | temperature baixa, prioriza consistência lógica e verificabilidade |
| `empatico` | salience baixo, prioriza contexto afetivo e linguagem segura |
| `pragmatico` | fronteira mínima de ações, resposta concisa |

O `GEAReflectionController` pontua cada candidato por empatia, segurança de runtime e simplicidade de ações, seleciona o vencedor e propõe micro-ajustes ao bridge (neuroplasticidade).

### Evolução e Introspecção

- **`IntrospectionEngine`**: analisa o dashboard de telemetria por sessão (felt, thought, decision, execution) e detecta gargalos — taxa de falha, backlog de aprovações, tendência de surpresa.
- **`EvolutionProposer`**: converte diagnósticos em `EvolutionDirective` (ajuste de parâmetro ou instrução de prompt) para revisão do operador.
- **`JsonlEvolutionArchive`**: persiste diretivas pendentes e aplicadas entre sessões.

### Runtime seguro

`StrictLambdaRuntime` valida cada ação contra um registro de ferramentas tipadas. Ações desconhecidas são rejeitadas imediatamente. O `AgentExecutionEngine` faz retry automático com feedback de crítica do `HeuristicVerifier` (loop de reparo com revisão guiada).

### Self-Model e Workspace

- **`CognitiveWorkspace`**: contexto compartilhado para um turno — inclui estado da sessão, snapshot de capacidades e diretivas ativas.
- **`build_self_model`**: gera um `CognitiveArchitectureMap` com componentes, conexões e superfície de adaptação do agente, usado para introspecção e exposição via API.

---

## Estrutura de módulos

```
src/calosum/
├── shared/
│   ├── ports.py              # Protocols injetáveis
│   ├── types.py              # Dataclasses centrais
│   ├── schemas.py            # Schemas da API
│   └── async_utils.py        # maybe_await, run_sync
│
├── domain/
│   ├── orchestrator.py       # CalosumAgent — pipeline principal
│   ├── bridge.py             # CognitiveTokenizer
│   ├── right_hemisphere.py   # RightHemisphereJEPA (heurístico)
│   ├── left_hemisphere.py    # LeftHemisphereLogicalSLM
│   ├── memory.py             # DualMemorySystem (episódica + semântica)
│   ├── persistent_memory.py  # JSONL / Qdrant / DuckDB
│   ├── runtime.py            # StrictLambdaRuntime
│   ├── runtime_dsl.py        # DSL de programas lambda tipados
│   ├── metacognition.py      # GEAReflectionController, variantes cognitivas
│   ├── verifier.py           # HeuristicVerifier
│   ├── telemetry.py          # CognitiveTelemetryBus
│   ├── event_bus.py          # InternalEventBus
│   ├── agent_execution.py    # AgentExecutionEngine (retry/repair)
│   ├── workspace.py          # CognitiveWorkspace
│   ├── self_model.py         # Snapshot de capacidades e auto-modelo
│   ├── evolution.py          # EvolutionProposer + JsonlEvolutionArchive
│   ├── introspection.py      # IntrospectionEngine
│   └── multiagent.py         # Raciocínio multi-papel (Planner/Executor/Verifier)
│
├── adapters/
│   ├── right_hemisphere_hf.py       # Percepção via HuggingFace (MiniLM)
│   ├── active_inference.py          # Surprise via energia livre
│   ├── llm_failover.py              # Roteamento multi-provedor com cooldown
│   ├── llm_qwen.py                  # Integração Qwen
│   ├── llm_payloads.py              # Formatação de output estruturado
│   ├── text_embeddings.py           # OpenAI / HuggingFace / lexical (fallback)
│   ├── memory_qdrant.py             # Qdrant vector store
│   ├── action_runtime.py            # Adaptadores de execução de ações
│   ├── bridge_store.py              # Persistência do estado do bridge
│   ├── telemetry_otlp.py            # Exportador OTLP
│   ├── knowledge_graph_nanorag.py   # NanoGraphRAG (opcional)
│   ├── channel_telegram.py          # Canal Telegram
│   ├── night_trainer.py             # Base de aprendizado contínuo
│   ├── night_trainer_dspy.py        # Auto-melhoria via DSPy
│   └── night_trainer_lora.py        # Fine-tuning LoRA noturno
│
├── bootstrap/
│   ├── factory.py    # CalosumAgentBuilder — injeção de dependência
│   ├── settings.py   # InfrastructureSettings
│   ├── api.py        # Servidor FastAPI
│   └── cli.py        # Entrypoints CLI
│
└── harness_checks.py   # Governança arquitetural (verificação AST)
```

---

## Instalação

**Requisitos**: Python 3.11+, Node.js 18+ (apenas para a UI)

```bash
pip install -r requirements.txt

# Verificar integridade arquitetural
PYTHONPATH=src python3 -m calosum.harness_checks
```

---

## Uso rápido

```bash
# Chat interativo
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
```

---

## Perfis de infraestrutura

Controlados pela variável `CALOSUM_INFRA_PROFILE`:

| Perfil | Comportamento |
|--------|---------------|
| `ephemeral` (padrão) | Tudo em RAM. Sem persistência. Ideal para testes. |
| `persistent` | Arquivos JSONL em `.calosum-runtime/`. Desenvolvimento local. |
| `docker` | Qdrant + OTLP + Jaeger. Ambiente produção-like. |

O bootstrap nunca falha duro por infraestrutura ausente: Qdrant → JSONL → RAM, HuggingFace → JEPA heurístico, embeddings remotos → lexical determinístico.

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

### Roteamento por papel cognitivo

| Variável | Papel |
|----------|-------|
| `CALOSUM_PERCEPTION_MODEL` | Hemisfério direito |
| `CALOSUM_REASON_MODEL` | Hemisfério esquerdo |
| `CALOSUM_REFLECTION_MODEL` | GEA reflection scoring |
| `CALOSUM_VERIFIER_MODEL` | Verificador |

### Infraestrutura

| Variável | Descrição |
|----------|-----------|
| `CALOSUM_VECTORDB_URL` | Qdrant (ex: `http://localhost:6333`) |
| `CALOSUM_EMBEDDING_ENDPOINT` | Serviço de embeddings |
| `CALOSUM_EMBEDDING_API_KEY` | Chave de embeddings |
| `CALOSUM_EMBEDDING_MODEL` | Modelo (ex: `text-embedding-3-small`) |
| `CALOSUM_MEMORY_DIR` | Diretório de memória persistente |
| `CALOSUM_DUCKDB_PATH` | Banco de dados do grafo semântico |
| `CALOSUM_OTEL_COLLECTOR_ENDPOINT` | Coletor OTLP (ex: `http://localhost:4318`) |
| `CALOSUM_JAEGER_UI_URL` | Interface Jaeger (ex: `http://localhost:16686`) |
| `CALOSUM_TELEGRAM_BOT_TOKEN` | Token do bot Telegram |

---

## Docker

```bash
docker compose -f deploy/docker-compose.yml up --build -d
```

| Serviço | Porta | Descrição |
|---------|-------|-----------|
| `orchestrator` | 8000 | API FastAPI do Calosum |
| `qdrant` | 6333 | Banco de vetores |
| `otel-collector` | 4317 / 4318 | Coletor OpenTelemetry |
| `jaeger` | 16686 | UI de rastreamento distribuído |

---

## UI de telemetria

Interface React para visualização do ciclo cognitivo em tempo real: timeline de eventos (felt, thought, decision, execution, reflection), scoreboard de variantes GEA e métricas de awareness de sessão.

```bash
cd ui && npm run dev     # desenvolvimento
npm run build            # produção
```

**Stack**: React 19, TypeScript, Vite, Tailwind CSS 4, lucide-react.

---

## Testes

```bash
# Suite completa
PYTHONPATH=src python3 -m unittest discover -s tests -t .

# Arquivo específico
PYTHONPATH=src python3 -m unittest tests.test_pipeline

# Método específico
PYTHONPATH=src python3 -m unittest tests.test_runtime.TestStrictLambdaRuntime.test_reject_unknown_action

# Governança arquitetural
PYTHONPATH=src python3 -m calosum.harness_checks
```

23 arquivos de teste cobrindo pipeline cognitivo, group turns, runtime seguro, memória dual, API, CLI, embeddings, telemetria OTLP, introspecção e auto-modelo.

---

## Desenvolvimento

### Adicionando uma nova integração

1. Defina um Protocol em `shared/ports.py`.
2. Implemente o adapter em `adapters/` atrás desse Protocol.
3. Registre no `CalosumAgentBuilder` em `bootstrap/factory.py`.
4. O domain não deve saber da existência do adapter.

### Regras de camada (verificadas automaticamente)

- `domain/` nunca importa de `adapters/` ou `bootstrap/`.
- Nenhum SDK externo (torch, transformers, httpx, openai) no `domain/` ou `shared/`.
- Limite de 400 linhas por arquivo.
- Mudanças em mais de um subsistema requerem plano em `docs/exec-plans/active/` (mover para `completed/` ao concluir).
- Dívida técnica registrada em `docs/exec-plans/tech-debt-tracker.md`.
