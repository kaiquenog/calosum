# Calosum

Calosum e um runtime experimental para agentes cognitivos neuro-simbolicos com arquitetura de duplo hemisferio, telemetria cognitiva, memoria persistente opcional e governanca arquitetural mecanica.

O repositorio evoluiu bastante desde a versao original. Hoje, o ponto de entrada principal do agente executa um fluxo de turno linear e observavel, com fallbacks explicitos entre adapters e uma camada de bootstrap responsavel por compor toda a infraestrutura.

## Comece Pelos Docs

O `README` e um guia de onboarding. A documentacao versionada do projeto fica em [`docs/index.md`](docs/index.md).

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): camadas, fronteiras e bootstrap
- [`docs/INFRASTRUCTURE.md`](docs/INFRASTRUCTURE.md): perfis, variaveis de ambiente e Docker
- [`docs/RELIABILITY.md`](docs/RELIABILITY.md): retries, telemetria e fallbacks
- [`docs/QUALITY_SCORE.md`](docs/QUALITY_SCORE.md): estado atual por area
- [`docs/PLANS.md`](docs/PLANS.md): quando abrir planos versionados

## O Que O Projeto Entrega Hoje

- Pipeline cognitivo com memoria, percepcao, bridge, planejamento, execucao e verificacao
- Arquitetura `Ports and Adapters` com regras de importacao verificadas por `harness_checks.py`
- CLI local para chat, turnos isolados, consolidacao de memoria, idle foraging e Night Trainer
- API FastAPI com rotas de chat, telemetria, introspeccao e estado do sistema
- UI React para explorar telemetria de sessoes
- Fallbacks explicitos para hemisferio direito, hemisferio esquerdo, embeddings, memoria vetorial e exportacao OTLP
- Integracoes opcionais com Qdrant, MCP, Telegram, OTLP e stack local de modelos

## Arquitetura Atual

O caminho principal de `process_turn` hoje e linear:

```text
UserTurn
  -> MemorySystem
  -> InputPerception (right hemisphere)
  -> CognitiveTokenizer / bridge
  -> ActionPlanner (left hemisphere)
  -> ActionRuntime
  -> Verifier
  -> Telemetry
  -> Awareness loop
  -> Persistencia de workspace e episodio
```

Isso acontece em [`src/calosum/domain/agent/orchestrator.py`](src/calosum/domain/agent/orchestrator.py).

Embora o repositorio ainda mantenha componentes de metacognicao, reflexao e evolucao, o fluxo operacional padrao do turno foi simplificado para priorizar previsibilidade, observabilidade e degradacao graciosa.

### Camadas

- `shared/`: tipos, `Protocol`s, serializacao e utilitarios puros
- `domain/`: logica cognitiva, memoria, execucao, metacognicao e telemetria
- `adapters/`: implementacoes concretas para LLMs, percepcao, memoria, ferramentas e canais
- `bootstrap/`: resolucao de settings, escolha de backends, factory e rotas HTTP

## Estrutura Do Repositorio

```text
src/calosum/
  shared/
    models/
    utils/
  domain/
    agent/
    cognition/
    execution/
    infrastructure/
    memory/
    metacognition/
  adapters/
    bridge/
    communication/
    execution/
    experience/
    hemisphere/
    knowledge/
    llm/
    memory/
    night_trainer/
    perception/
    tools/
  bootstrap/
    entry/
    infrastructure/
    routers/
    wiring/
ui/
docs/
tests/
deploy/
```

## Instalacao

Requisitos minimos:

- Python 3.11+
- Node.js para a UI
- `npm` para build e lint do frontend

Exemplo de ambiente local:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Se voce for usar stack local de modelos, instale os extras opcionais:

```bash
python -m pip install -e .[local]
```

Para desenvolvimento dentro do repositorio, os comandos abaixo assumem `PYTHONPATH=src`.

## Uso Rapido

### Chat interativo

```bash
PYTHONPATH=src ./.venv/bin/python3 -m calosum chat --session-id terminal-session
```

### Processar um turno unico

```bash
PYTHONPATH=src ./.venv/bin/python3 -m calosum run-turn \
  --session-id demo \
  --text "Preciso organizar um plano tecnico curto e seguro."
```

### Consolidar memoria

```bash
PYTHONPATH=src ./.venv/bin/python3 -m calosum sleep
```

### Rodar Night Trainer

```bash
PYTHONPATH=src ./.venv/bin/python3 -m calosum optimize-prompts --backend dspy
```

### Idle foraging

```bash
PYTHONPATH=src ./.venv/bin/python3 -m calosum idle
```

## API HTTP

Suba a API:

```bash
PYTHONPATH=src ./.venv/bin/python3 -m calosum.bootstrap.entry.api
```

Rotas principais:

- `GET /health`
- `GET /ready`
- `POST /v1/chat/completions`
- `GET /v1/chat/sse`
- `GET /v1/system/info`
- `GET /v1/system/capabilities`
- `GET /v1/system/state`
- `GET /v1/system/awareness`
- `POST /v1/system/introspect`
- `POST /v1/system/idle`
- `GET /v1/telemetry/dashboard`
- `GET /v1/telemetry/dashboard/{session_id}`
- `POST /v1/telemetry/query`

Exemplo:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo",
    "text": "Resuma os riscos operacionais desta tarefa em 3 pontos."
  }'
```

## UI De Telemetria

```bash
cd ui
npm install
npm run dev
```

A UI consome os endpoints de telemetria e sistema expostos pela API do backend.

## Perfis E Modos

As configuracoes principais vivem em [`src/calosum/bootstrap/infrastructure/settings.py`](src/calosum/bootstrap/infrastructure/settings.py) e estao detalhadas em [`docs/INFRASTRUCTURE.md`](docs/INFRASTRUCTURE.md).

As tres variaveis mais importantes sao:

- `CALOSUM_INFRA_PROFILE=ephemeral|persistent|docker`
- `CALOSUM_MODE=api|local`
- `CALOSUM_DEPENDENCY_MODE=auto|api|local`

Resumo pratico:

- `ephemeral`: tudo em memoria; melhor para testes e execucao rapida
- `persistent`: persiste memoria, telemetria e artefatos em `.calosum-runtime/`
- `docker`: usa a stack em `deploy/docker-compose.yml`
- `mode=api`: privilegia endpoints externos e evita stack local de treino pesado
- `mode=local`: habilita cenarios com runtime/modelos locais

## Variaveis De Ambiente Mais Uteis

```bash
CALOSUM_INFRA_PROFILE=persistent
CALOSUM_MODE=api
CALOSUM_DEPENDENCY_MODE=auto

CALOSUM_LEFT_ENDPOINT=http://localhost:8001/v1
CALOSUM_LEFT_MODEL=Qwen/Qwen-3.5-9B-Instruct
CALOSUM_LEFT_API_KEY=...

CALOSUM_RIGHT_BACKEND=huggingface
CALOSUM_VECTORDB_URL=http://localhost:6333
CALOSUM_OTEL_COLLECTOR_ENDPOINT=http://localhost:4318

CALOSUM_MCP_ENABLED=true
CALOSUM_MCP_SERVERS={"docs":"http://localhost:9000/mcp"}
```

Tambem existem configuracoes para:

- embeddings: `CALOSUM_EMBEDDING_*`
- failover do hemisferio esquerdo: `CALOSUM_LEFT_FALLBACK_*`
- bridge/fusao: `CALOSUM_BRIDGE_BACKEND`, `CALOSUM_FUSION_*`
- Telegram: `TELEGRAM_BOT_TOKEN`, `CALOSUM_TELEGRAM_*`
- vault seguro para tools: `CALOSUM_VAULT_*`

## Docker

```bash
docker compose -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml ps
```

O perfil `docker` foi desenhado para subir API, Qdrant e observabilidade. A UI continua sendo rodada localmente em `ui/`.

## Qualidade E Governanca

Checklist minimo antes de mudar comportamento estrutural:

```bash
PYTHONPATH=src ./.venv/bin/python3 -m calosum.harness_checks
PYTHONPATH=src ./.venv/bin/python3 -m unittest discover -s tests -t .
cd ui && npm run lint && npm run build
```

O harness valida, entre outras coisas:

- fronteiras entre camadas
- modulos sem regra em `MODULE_RULES`
- tamanho maximo de modulo
- consistencia basica da documentacao de governanca

## Notas Para Quem Vai Contribuir

- O bootstrap e o unico lugar autorizado a acoplar `domain` e `adapters`
- Novas integracoes externas devem entrar atras de `Protocol`s em `shared`
- Mudancas cross-cutting exigem plano versionado em `docs/exec-plans/active/`
- Regras recorrentes devem virar checks mecanicos no harness

## Referencias Rapidas

- [`docs/index.md`](docs/index.md)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/INFRASTRUCTURE.md`](docs/INFRASTRUCTURE.md)
- [`docs/RELIABILITY.md`](docs/RELIABILITY.md)
- [`docs/QUALITY_SCORE.md`](docs/QUALITY_SCORE.md)
- [`docs/production-roadmap.md`](docs/production-roadmap.md)
