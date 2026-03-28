# Infraestrutura

O sistema `calosum` é agnóstico de ambiente, orquestrando dependências através da classe `InfrastructureSettings` e `CalosumAgentBuilder`.

## Variáveis de Ambiente

As configurações principais podem ser passadas via `.env` ou exportadas no terminal:

- `CALOSUM_INFRA_PROFILE`: Define o comportamento global (`ephemeral`, `persistent`, `docker`).
- `CALOSUM_VECTORDB_URL`: Endpoint do Qdrant (ex: `http://localhost:6333`).
- `CALOSUM_LEFT_ENDPOINT`: URL base da API compatível com OpenAI (ex: vLLM, Ollama).
- `CALOSUM_LEFT_MODEL`: Nome do modelo (ex: `Qwen/Qwen-3.5-9B-Instruct`).
- `CALOSUM_API_PORT`: Porta para a API REST/SSE (padrão: 8000).
- `CALOSUM_VAULT_*`: Variáveis prefixadas com este padrão são automaticamente injetadas de forma segura no `ActionRuntime` (ex: `CALOSUM_VAULT_GITHUB_TOKEN`).

## Perfis de Execução

### 1. Ephemeral (Padrão)
`CALOSUM_INFRA_PROFILE=ephemeral`
- A memória vetorial e a telemetria rodam totalmente na memória RAM (arrays de Python).
- Nada é salvo no disco. Ideal para rodar a suíte de testes rápidos.
- Se a stack opcional de `transformers`/`sentence-transformers` não estiver disponível, o bootstrap faz fallback para o hemisfério direito heurístico sem abortar a inicialização.

### 2. Persistent
`CALOSUM_INFRA_PROFILE=persistent`
- Salva dados na pasta local `.calosum-runtime/`.
- Os eventos de telemetria vão para `.calosum-runtime/telemetry/events.jsonl`.
- Se o Qdrant não for configurado, a memória vetorial salva localmente em formato `.jsonl` serializado.

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
