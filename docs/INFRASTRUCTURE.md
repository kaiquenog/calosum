# Infraestrutura

O sistema `calosum` é agnóstico de ambiente, orquestrando dependências através da classe `InfrastructureSettings` e `CalosumAgentBuilder`.

## Variáveis de Ambiente

As configurações principais podem ser passadas via `.env` ou exportadas no terminal:

- `CALOSUM_INFRA_PROFILE`: Define o comportamento global (`ephemeral`, `persistent`, `docker`).
- `CALOSUM_VECTORDB_URL`: Endpoint do Qdrant (ex: `http://localhost:6333`).
- `CALOSUM_LEFT_ENDPOINT`: URL base da API compatível com OpenAI (ex: vLLM, Ollama).
- `CALOSUM_LEFT_MODEL`: Nome do modelo (ex: `Qwen/Qwen-3.5-9B-Instruct`).
- `CALOSUM_LEFT_PROVIDER`: Opcional. Forca o modo `openai_responses`, `openai_chat` ou `openai_compatible_chat`.
- `CALOSUM_LEFT_REASONING_EFFORT`: Opcional. Quando o provider e OpenAI Responses, envia `reasoning.effort` para modelos compatíveis.
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

### 2. Persistent
`CALOSUM_INFRA_PROFILE=persistent`
- Salva dados na pasta local `.calosum-runtime/`.
- Os eventos de telemetria vão para `.calosum-runtime/telemetry/events.jsonl`.
- Se o Qdrant não for configurado, a memória vetorial salva localmente em formato `.jsonl` serializado.
- Sem configuracao explicita, a API e o comando `python3 -m calosum.bootstrap.cli chat` passam a adotar este modo localmente para permitir que a UI consulte a mesma telemetria entre processos.
- Quando o Qdrant estiver ativo, o adapter de memoria usa embeddings configuraveis. Sem endpoint externo ou stack local de Sentence Transformers, ele cai para um embedding lexical deterministico para manter a busca vetorial funcional.

### 3. Docker
`CALOSUM_INFRA_PROFILE=docker`
- Utiliza a stack definida em `deploy/docker-compose.yml`.
- Sobe Qdrant, Jaeger (OpenTelemetry) e a API do Orquestrador.
- **Machine Learning**: Os modelos da biblioteca `sentence-transformers` e `torch` baixam os pesos durante a criação da imagem e rodam na V-NET do container.
- **Frontend UI:** Para rodar a interface de telemetria em React, inicie-a localmente usando `npm run dev` dentro do diretório `ui/`.

## Rotinas Assíncronas (Sleep Mode)

A infraestrutura prevê a execução de consolidação noturna e neuroplasticidade. O arquivo `adapters/night_trainer.py` pode ser executado isoladamente (como um CronJob no container ou na máquina host) para ler o `.jsonl` extraído do Qdrant e aplicar um fine-tuning via **PEFT/LoRA**.

## Observabilidade (OTEL)

A telemetria cognitiva está preparada para exportar logs `JSONL` compatíveis com o formato OpenTelemetry. Quando o perfil Docker é ativado, esses logs podem ser consumidos por coletores (ex: `otel-collector`) para roteamento a visualizadores de rastreamento distribuído (Jaeger).

Adicionalmente, os dados de telemetria da sessão estão disponíveis em tempo real através da API REST (`/v1/telemetry/dashboard/{session_id}`) e consumidos ativamente pelo painel UI React.

## Fluxo Local Recomendado

Para observar um chat local na UI:

1. Suba a API do Calosum.
2. Inicie o chat com `python3 -m calosum.bootstrap.cli chat`.
3. Abra a UI apontando para a API.
4. Observe a sessao `terminal-session`, que e o identificador padrao do REPL e do painel.

Nesse fluxo, a CLI grava a telemetria em `.calosum-runtime/telemetry/events.jsonl` e a API reidrata o dashboard a partir do mesmo arquivo.

## OpenAI

- Se `CALOSUM_LEFT_ENDPOINT` apontar para `https://api.openai.com/v1`, o adapter autodetecta OpenAI oficial e usa `Responses API` com Structured Outputs.
- Se `CALOSUM_VECTORDB_URL` estiver configurado e nenhum backend de embedding separado for informado, o builder reutiliza o endpoint OpenAI oficial para embeddings e deriva `text-embedding-3-small`.
- Se voce apontar para um endpoint local no formato OpenAI-compatible, como Ollama ou vLLM, o adapter mantem `chat/completions`.
- Para workloads mais inteligentes e pesados, prefira `gpt-5.4`. Para menor custo e latencia, prefira `gpt-5-mini`.
