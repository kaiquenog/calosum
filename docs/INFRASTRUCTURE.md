# Infraestrutura

O sistema `calosum` é agnóstico de ambiente, orquestrando dependências através da classe `InfrastructureSettings` e `CalosumAgentBuilder`.

## Variáveis de Ambiente

As configurações principais podem ser passadas via `.env` ou exportadas no terminal:

- `CALOSUM_INFRA_PROFILE`: Define o comportamento global (`ephemeral`, `persistent`, `docker`).
- `CALOSUM_MODE`: Define o modo de execução (`api`, `local`).
  - `api`: usa endpoint externo de LLM; Night Trainer fica restrito a DSPy.
  - `local`: usa stack local; endpoint externo do hemisfério esquerdo é ignorado no bootstrap padrão.
- `CALOSUM_DEPENDENCY_MODE`: Compatibilidade de instalação (`auto`, `api`, `local`). Deve ser coerente com `CALOSUM_MODE`.
- `CALOSUM_VECTORDB_URL`: Endpoint do Qdrant (ex: `http://localhost:6333`).
- `CALOSUM_LEFT_ENDPOINT`: URL base da API compatível com OpenAI (ex: vLLM, Ollama).
- `CALOSUM_LEFT_MODEL`: Nome do modelo (ex: `Qwen/Qwen-3.5-9B-Instruct`).
- `CALOSUM_LEFT_PROVIDER`: Opcional. Forca o modo `openai_responses`, `openai_chat` ou `openai_compatible_chat`.
- `CALOSUM_LEFT_REASONING_EFFORT`: Opcional. Quando o provider e OpenAI Responses, envia `reasoning.effort` para modelos compatíveis.
- `CALOSUM_LEFT_FALLBACK_ENDPOINT`: Opcional. Segundo endpoint do hemisferio esquerdo para failover automatico.
- `CALOSUM_LEFT_FALLBACK_MODEL`: Opcional. Modelo do endpoint de fallback. Se omitido, reutiliza o modelo primario.
- `CALOSUM_LEFT_FALLBACK_PROVIDER`: Opcional. Provider do endpoint de fallback.
- `CALOSUM_LEFT_FALLBACK_REASONING_EFFORT`: Opcional. `reasoning.effort` do endpoint de fallback.
- `CALOSUM_LEFT_FALLBACK_API_KEY`: Opcional. Chave especifica do endpoint de fallback.
- `CALOSUM_LEFT_BACKEND`: Opcional. Forca o backend do hemisferio esquerdo (`rlm` ou `qwen`).
- `CALOSUM_LEFT_RLM_RUNTIME_COMMAND`: Opcional. Comando local para executar o runtime RLM oficial.
- `CALOSUM_LEFT_RLM_PATH`: Opcional. Caminho do modelo/artefato RLM local.
- `CALOSUM_LEFT_RLM_MAX_DEPTH`: Opcional. Profundidade maxima de recursao do backend RLM.
- `CALOSUM_RIGHT_BACKEND`: Opcional. Seleciona backend do hemisferio direito (`auto`, `vjepa21`, `vljepa`, `jepars`, `huggingface`).
- `CALOSUM_RIGHT_MODEL_PATH`: Opcional. Caminho para artefatos locais JEPA/ONNX.
- `CALOSUM_RIGHT_ACTION_CONDITIONED`: Opcional. Liga/desliga condicionamento por acao no preditor direito.
- `CALOSUM_RIGHT_HORIZON`: Opcional. Horizonte de predicao do hemisferio direito.
- `CALOSUM_RIGHT_JEPARS_BINARY`: Opcional. Binario do backend Rust `jepa-rs`.
- `CALOSUM_BRIDGE_BACKEND`: Opcional. Seleciona a fusao do corpus caloso (`heuristic` ou `cross_attention`).
- `CALOSUM_LEFT_PROMPT_PATH`: Opcional. Caminho para template externo de prompt base do hemisferio esquerdo (prompt-as-data).
- `CALOSUM_FUSION_ENABLED`: Opcional. Liga/desliga fusao semantica JEPA+LLM no hemisferio esquerdo.
- `CALOSUM_FUSION_CANDIDATES`: Opcional. Numero de candidatos gerados por turno quando fusao estiver ativa (padrao: 3).
- `CALOSUM_FUSION_SELECTION_MODE`: Opcional. Modo de selecao (`guided` para JEPA-guided, `random` para controle B).
- `CALOSUM_FUSION_UNCERTAINTY_THRESHOLD`: Opcional. Trigger maximo de incerteza JEPA para ativar custo extra da fusao (padrao: 0.5).
- `CALOSUM_MCP_ENABLED`: Opcional. Liga suporte a chamadas MCP no runtime (`call_mcp_tool`).
- `CALOSUM_MCP_SERVERS`: Opcional. JSON `{ "nome":"http://host:porta/mcp" }` com endpoints MCP via HTTP.
- `CALOSUM_MCP_ALLOWLIST`: Opcional. Lista CSV de servidores MCP permitidos (controle de seguranca).
- `CALOSUM_GEA_SHARING_ENABLED`: Opcional. Ativa experience sharing persistente no ReflectionController.
- `CALOSUM_GEA_EXPERIENCE_STORE_PATH`: Opcional. Caminho do banco SQLite de experiencias do GEA.
- `CALOSUM_NIGHT_TRAINER_BACKEND`: Opcional. Seleciona o backend do ciclo noturno (`auto`, `dspy`, `opro_lite`, `lora`, `qlora`).
- `CALOSUM_EMBEDDING_ENDPOINT`: Opcional. Endpoint para embeddings usados pelo adapter Qdrant.
- `CALOSUM_EMBEDDING_MODEL`: Opcional. Modelo de embedding. Em OpenAI oficial, o default e `text-embedding-3-small`.
- `CALOSUM_EMBEDDING_PROVIDER`: Opcional. Forca `openai`, `openai_compatible`, `huggingface` ou `lexical`.
- `CALOSUM_EMBEDDING_API_KEY`: Opcional. Chave especifica para o backend de embeddings.
- `CALOSUM_API_PORT`: Porta para a API REST/SSE (padrão: 8000).
- `CALOSUM_VAULT_*`: Variáveis prefixadas com este padrão são automaticamente injetadas de forma segura no `ActionRuntime` (ex: `CALOSUM_VAULT_GITHUB_TOKEN`).

## Perfis de Execução

### 1. Ephemeral (Padrão)
`CALOSUM_INFRA_PROFILE=ephemeral`
- A memória vetorial e a telemetria rodam totalmente na memória RAM (arrays de Python).
- Nada é salvo no disco. Ideal para rodar a suíte de testes rápidos.
- Se a stack opcional de `transformers`/`sentence-transformers` não estiver disponível, o bootstrap faz fallback para o hemisfério direito heurístico sem abortar a inicialização.
- Se voce iniciar explicitamente API ou chat com este perfil configurado, a UI nao enxergara eventos de outros processos porque a telemetria fica isolada em memoria.
- A fusao JEPA+LLM fica desabilitada por padrao para evitar custo extra de chamadas LLM (pode ser habilitada via `CALOSUM_FUSION_ENABLED=true`).

### 2. Persistent
`CALOSUM_INFRA_PROFILE=persistent`
- Salva dados na pasta local `.calosum-runtime/`.
- Os eventos de telemetria vão para `.calosum-runtime/telemetry/events.jsonl`.
- Se o Qdrant não for configurado, a memória vetorial salva localmente em formato `.jsonl` serializado.
- Sem configuracao explicita, a API e o comando `python3 -m calosum chat` (ou `calosum chat` após `pip install -e .`) passam a adotar este modo localmente para permitir que a UI consulte a mesma telemetria entre processos.
- Quando o Qdrant estiver ativo, o adapter de memoria usa embeddings configuraveis. Sem endpoint externo ou stack local de Sentence Transformers, ele cai para um embedding lexical deterministico para manter a busca vetorial funcional.

### 3. Docker
`CALOSUM_INFRA_PROFILE=docker`
- Utiliza a stack definida em `deploy/docker-compose.yml`.
- Sobe Qdrant, Jaeger (OpenTelemetry) e a API do Orquestrador.
- **Machine Learning**: Os modelos da biblioteca `sentence-transformers` e `torch` baixam os pesos durante a criação da imagem e rodam na V-NET do container.
- **Frontend UI:** Para rodar a interface de telemetria em React, inicie-a localmente usando `npm run dev` dentro do diretório `ui/`.

## Rotinas Assíncronas (Sleep Mode)

A infraestrutura prevê a execução de consolidação noturna e neuroplasticidade. O arquivo `adapters/night_trainer.py` pode ser executado isoladamente (como um CronJob no container ou na máquina host) para ler o `.jsonl` extraído do Qdrant e compilar um artefato offline de prompt reutilizado pelo hemisfério esquerdo no dia seguinte. O backend padrão tenta `DSPy` quando a dependência está presente e cai para `OPRO-lite` quando o ambiente local não possui o otimizador.

O bootstrap também injeta um store de conhecimento em grafo persistido em `knowledge_graph.jsonl`. Quando `nano-graphrag` não está instalado, a recuperação semântica cai para um backend local `NetworkX` com expansão de subgrafo, mantendo o contrato de `knowledge_triples` consumido pelo hemisfério esquerdo.

## Observabilidade (OTEL)

A telemetria cognitiva exporta spans via HTTP/OTLP para o coletor configurado. O fluxo completo é:

```
Agente (OTLPHTTPTraceSink) → otel-collector:4318 → Jaeger:4317 → Jaeger UI:16686
```

### Configuração

A variável `CALOSUM_OTEL_COLLECTOR_ENDPOINT` define o endpoint do coletor OTLP (padrão em Docker: `http://otel-collector:4318`). O adapter `adapters/telemetry_otlp.py` exporta spans via `POST /v1/traces` usando HTTP+JSON.

**Fallback gracioso:** Se o coletor não estiver disponível, o adapter registra `WARNING: Failed to export OTLP trace` e continua sem lançar exceção — o agente nunca aborta por falha de telemetria.

### Validando traces OTLP (stack Docker completa)

```bash
# 1. Subir a stack de infraestrutura
docker compose -f deploy/docker-compose.yml up -d

# 2. Aguardar os containers ficarem saudáveis (~15s)
docker compose -f deploy/docker-compose.yml ps

# 3. Rodar um turno pelo CLI (local conectando ao collector em Docker)
CALOSUM_OTEL_COLLECTOR_ENDPOINT=http://localhost:4318 \
CALOSUM_INFRA_PROFILE=persistent \
PYTHONPATH=src .venv/bin/python3 -m calosum chat
# (envie uma mensagem e saia com Ctrl+C)

# 4. Abrir o Jaeger UI e verificar a trace
open http://localhost:16686
# Selecionar service: "calosum" → Find Traces
```

### Arquivo de configuração do coletor

`deploy/otel-collector-config.yaml` — define o pipeline:
- **Receiver:** `otlp` (gRPC:4317, HTTP:4318)
- **Processor:** `batch`
- **Exporters:** `debug` (log básico) + `otlp/jaeger` (para Jaeger via gRPC:4317)

Adicionalmente, os dados de telemetria da sessão estão disponíveis em tempo real através da API REST (`/v1/telemetry/dashboard/{session_id}`) e consumidos pelo painel UI React.


## Fluxo Local Recomendado

Para observar um chat local na UI:

1. Suba a API do Calosum.
2. Inicie o chat com `python3 -m calosum chat`.
3. Abra a UI apontando para a API.
4. Observe a sessao `terminal-session`, que e o identificador padrao do REPL e do painel.

Nesse fluxo, a CLI grava a telemetria em `.calosum-runtime/telemetry/events.jsonl` e a API reidrata o dashboard a partir do mesmo arquivo.

## OpenAI

- Se `CALOSUM_LEFT_ENDPOINT` apontar para `https://api.openai.com/v1`, o adapter autodetecta OpenAI oficial e usa `Responses API` com Structured Outputs.
- Se `CALOSUM_LEFT_FALLBACK_ENDPOINT` estiver definido, o bootstrap monta um adapter resiliente e tenta automaticamente o endpoint secundario quando o primario devolve erro estrutural ou falha operacional.
- Se `CALOSUM_VECTORDB_URL` estiver configurado e nenhum backend de embedding separado for informado, o builder reutiliza o endpoint OpenAI oficial para embeddings e deriva `text-embedding-3-small`.
- Se voce apontar para um endpoint local no formato OpenAI-compatible, como Ollama ou vLLM, o adapter mantem `chat/completions`.
- Para workloads mais inteligentes e pesados, prefira `gpt-5.4`. Para menor custo e latencia, prefira `gpt-5-mini`.

## Extensoes Cognitivas

- O runtime agora expoe `spawn_subordinate` para delegacao de subtarefas isoladas.
- Hooks/interceptors podem observar eventos de ciclo (`message_loop_start`, `after_perception`, `after_turn_execution`) e execucao de tool (`before_tool_execution`, `after_tool_execution`) sem acoplamento no dominio.
