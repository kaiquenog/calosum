# Calosum

Runtime experimental para agentes cognitivos neuro-simbólicos com arquitetura de **duplo hemisfério** (percepção latente + raciocínio LLM), telemetria cognitiva, memória persistente opcional e **governança arquitetural mecânica** (`harness_checks`).

O fluxo operacional padrão do turno é **linear e observável**, com fallbacks explícitos entre adapters e uma camada de **bootstrap** que compõe toda a infraestrutura.

# Proximos passos

A Ideia geral depois do projeto criar auto conciencia da sua estrtutura nao AGI mais sim estrutura arquitetural para que possa analisar sua telemetria e melhorar sua performance e sugerir melhorias de modulos se auto indexando, vamos usar RLM (Recursive Language Model) para isso, para testar cenarios provaveis de implementacao vou pedir a um especcialista de que ajude a implementar o GEA(Group-Evolving Agents: Open-Ended Self-Improvement via Experience Sharing)

---

## Índice

- [O que o projeto entrega](#o-que-o-projeto-entrega)
- [Arquitetura](#arquitetura)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Uso rápido (CLI)](#uso-rápido-cli)
- [API HTTP](#api-http)
- [UI de telemetria (opcional)](#ui-de-telemetria-opcional)
- [Perfis e variáveis de ambiente](#perfis-e-variáveis-de-ambiente)
- [Docker](#docker)
- [Testes, harness e CI](#testes-harness-e-ci)
- [Contribuindo](#contribuindo)
- [Documentação](#documentação)

---

## O que o projeto entrega

- Pipeline cognitivo com memória, percepção, *bridge*, planejamento, execução e verificação
- Arquitetura **Ports and Adapters** com regras de importação verificadas por `harness_checks.py`
- **CLI** local: chat, turnos isolados, cenários JSON, consolidação de memória (*sleep*), *idle foraging* e Night Trainer
- **API FastAPI**: chat, SSE, telemetria, introspeção e estado do sistema
- **UI React** (quando o diretório `ui/` existe no checkout) para explorar telemetria de sessões
- Fallbacks explícitos para hemisfério direito, hemisfério esquerdo, embeddings, memória vetorial e exportação OTLP
- Integrações opcionais: Qdrant, MCP, Telegram, OTLP e stack local de modelos (`torch` / `transformers`, etc.)

---

## Arquitetura

### Fluxo principal do turno

O caminho de `process_turn` é linear; a coordenação está em [`src/calosum/domain/agent/orchestrator.py`](src/calosum/domain/agent/orchestrator.py).

```text
UserTurn
  -> MemorySystem
  -> InputPerception (hemisfério direito)
  -> CognitiveTokenizer / bridge
  -> ActionPlanner (hemisfério esquerdo)
  -> ActionRuntime
  -> Verifier
  -> Telemetry
  -> Awareness loop
  -> Persistência de workspace e episódio
```

Componentes de metacognição, reflexão e evolução ainda existem no repositório; o fluxo padrão do turno prioriza previsibilidade, observabilidade e degradação graciosa.

Quando o branching multi-candidato está ativo, o contrato estável é `GroupTurnResult.selected_result -> AgentTurnResult`. O caminho de compatibilidade fica explícito em `/ready` e nos benchmarks de reflection.

### Camadas

| Camada | Papel |
|--------|--------|
| `shared/` | Tipos, `Protocol`s, serialização e utilitários puros |
| `domain/` | Lógica cognitiva, memória, execução, metacognição e telemetria (sem SDKs externos) |
| `adapters/` | Implementações concretas (LLM, percepção, memória, ferramentas, canais) |
| `bootstrap/` | Settings, factory, CLI, API e rotas HTTP — único ponto que acopla `domain` e `adapters` |

Detalhes: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Estrutura do repositório

```text
src/calosum/
  shared/          # tipos, ports, schemas
  domain/          # agente, cognição, execução, memória, metacognição
  adapters/        # LLM, hemisférios, bridge, tools, percepção, etc.
  bootstrap/       # entry (CLI/API), wiring, routers, settings
  harness_checks.py

docs/              # documentação versionada (índice em docs/index.md)
tests/             # domain, adapters, bootstrap, shared, integration
deploy/            # docker-compose e artefatos de deploy
ui/                # frontend opcional (React/Vite) — pode não existir em todos os checkouts
```

---

## Requisitos

- **Python 3.11+**
- **Node.js** e **npm** — apenas se for desenvolver ou buildar a UI em `ui/`
- **Docker** — opcional, para stack com Qdrant, API e observabilidade (ver [`deploy/docker-compose.yml`](deploy/docker-compose.yml))

---

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Para stack local de modelos (Torch, PEFT, transformers, etc.):

```bash
python -m pip install -e ".[local]"
```

Desenvolvimento no repositório costuma usar `PYTHONPATH=src` para executar módulos sem depender só do *entry point* instalado.

Após `pip install -e .`, ficam disponíveis os comandos de console:

- `calosum` — CLI (equivale a `python -m calosum`)
- `calosum-harness` — governança (`python -m calosum.harness_checks`)

---

## Uso rápido (CLI)

Com o ambiente ativado e, se necessário, `export PYTHONPATH=src`:

| Objetivo | Comando |
|----------|---------|
| Ajuda | `python -m calosum --help` |
| Chat interativo | `python -m calosum chat --session-id terminal-session` |
| Um turno | `python -m calosum run-turn --session-id demo --text "Sua mensagem."` |
| Cenário JSON | `python -m calosum run-scenario --help` |
| Consolidação de memória (*sleep*) | `python -m calosum sleep` |
| Night Trainer (ex.: DSPy) | `python -m calosum optimize-prompts --backend dspy` |
| *Idle foraging* | `python -m calosum idle` |

Exemplos com caminho explícito ao interpretador:

```bash
PYTHONPATH=src ./.venv/bin/python3 -m calosum chat --session-id terminal-session
```

```bash
PYTHONPATH=src ./.venv/bin/python3 -m calosum run-turn \
  --session-id demo \
  --text "Preciso organizar um plano técnico curto e seguro."
```

---

## API HTTP

Subir o servidor (padrão: porta **8000**, configurável com `CALOSUM_API_PORT`):

```bash
PYTHONPATH=src ./.venv/bin/python3 -m calosum.bootstrap.entry.api
```

Rotas principais:

- `GET /health`, `GET /ready`
- `POST /v1/chat/completions`
- `GET /v1/chat/sse`
- `GET /v1/system/info`, `GET /v1/system/capabilities`, `GET /v1/system/state`, `GET /v1/system/awareness`
- `POST /v1/system/introspect`, `POST /v1/system/idle`
- `GET /v1/telemetry/dashboard`, `GET /v1/telemetry/dashboard/{session_id}`, `POST /v1/telemetry/query`

Exemplo:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo",
    "text": "Resuma os riscos operacionais desta tarefa em 3 pontos."
  }'
```

`GET /ready` agora também devolve `operational_budgets` por componente e o `turn_contract` vigente entre `AgentTurnResult` e `GroupTurnResult`.

---

## UI de telemetria (opcional)

Se o diretório **`ui/`** existir no seu clone (React, Vite, Tailwind):

```bash
cd ui
npm install
npm run dev
```

O painel consome os endpoints de telemetria e sistema da API. **Se `ui/` não estiver presente**, use a API e os endpoints de telemetria acima (por exemplo o dashboard em `/v1/telemetry/dashboard`).

---

## Perfis e variáveis de ambiente

As configurações centrais vivem em [`src/calosum/bootstrap/infrastructure/settings.py`](src/calosum/bootstrap/infrastructure/settings.py). Referência completa: [`docs/INFRASTRUCTURE.md`](docs/INFRASTRUCTURE.md).

Três variáveis guiam o comportamento global:

| Variável | Valores típicos | Efeito resumido |
|----------|-----------------|-----------------|
| `CALOSUM_INFRA_PROFILE` | `ephemeral`, `persistent`, `docker` | Onde e como persistem memória, telemetria e dependências |
| `CALOSUM_MODE` | `api`, `local` | Privilegia endpoints externos vs. stack local |
| `CALOSUM_DEPENDENCY_MODE` | `auto`, `api`, `local` | Alinhamento de instalação com o modo de execução |

Budgets operacionais opcionais por componente:

- `CALOSUM_RIGHT_BUDGET_CPU_CORES`, `CALOSUM_RIGHT_BUDGET_MEMORY_MB`
- `CALOSUM_LEFT_BUDGET_CPU_CORES`, `CALOSUM_LEFT_BUDGET_MEMORY_MB`
- `CALOSUM_BRIDGE_BUDGET_CPU_CORES`, `CALOSUM_BRIDGE_BUDGET_MEMORY_MB`
- `CALOSUM_BUDGET_CPU_CORES`, `CALOSUM_BUDGET_MEMORY_MB` como limite global de fallback

Exemplo mínimo para desenvolvimento com API externa e persistência local:

```bash
export CALOSUM_INFRA_PROFILE=persistent
export CALOSUM_MODE=api
export CALOSUM_DEPENDENCY_MODE=auto

export CALOSUM_LEFT_ENDPOINT=http://localhost:8001/v1
export CALOSUM_LEFT_MODEL=Qwen/Qwen-3.5-9B-Instruct
export CALOSUM_LEFT_API_KEY=...

export CALOSUM_RIGHT_BACKEND=huggingface
export CALOSUM_VECTORDB_URL=http://localhost:6333
export CALOSUM_OTEL_COLLECTOR_ENDPOINT=http://localhost:4318

export CALOSUM_MCP_ENABLED=true
export CALOSUM_MCP_SERVERS='{"docs":"http://localhost:9000/mcp"}'
```

Há também variáveis para embeddings (`CALOSUM_EMBEDDING_*`), failover do hemisfério esquerdo (`CALOSUM_LEFT_FALLBACK_*`), bridge/fusão (`CALOSUM_BRIDGE_BACKEND`, `CALOSUM_FUSION_*`), Telegram e *vault* de credenciais para tools (`CALOSUM_VAULT_*`).

---

## Docker

```bash
docker compose -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml ps
```

O perfil `docker` costuma incluir API, Qdrant e observabilidade. A UI, quando existir, continua sendo executada no host com `npm run dev` em `ui/`.

---

## Testes, harness e CI

Checklist mínimo antes de mudanças estruturais ou de abrir PR:

```bash
PYTHONPATH=src ./.venv/bin/python3 -m calosum.harness_checks
PYTHONPATH=src ./.venv/bin/python3 -m unittest discover -s tests -t .
```

Se houver alterações no frontend:

```bash
cd ui && npm run lint && npm run build
```

O *harness* valida, entre outras coisas: fronteiras entre camadas, regras `MODULE_RULES`, tamanho máximo de módulo e consistência básica da documentação de governança.

O CI em [`.github/workflows/ci.yml`](.github/workflows/ci.yml) inclui *harness*, Ruff, mypy estrito em arquivos alterados, testes com *gate* de cobertura, integração e *benchmark gate* (ver [`docs/benchmarks/baseline.md`](docs/benchmarks/baseline.md)).

---

## Contribuindo

- O **bootstrap** é o único lugar autorizado a acoplar `domain` e `adapters`
- Novas integrações externas devem entrar atrás de `Protocol`s em `shared` (por exemplo `shared/ports.py`)
- Mudanças *cross-cutting* exigem plano versionado em [`docs/exec-plans/active/`](docs/exec-plans/active/) (ver [`docs/PLANS.md`](docs/PLANS.md))
- Regras recorrentes devem virar checks no `harness_checks.py` quando fizer sentido
- O ponto de entrada do agente é o orquestrador; **CLI** e **API** não devem vazar contexto indevido para o `domain`

Mais detalhes operacionais para agentes e desenvolvedores: [`AGENTS.md`](AGENTS.md).

---

## Documentação

- Contrato `jepa-rs`: [`docs/components/right-hemisphere-jepa-rs-contract.md`](docs/components/right-hemisphere-jepa-rs-contract.md)

| Documento | Conteúdo |
|-----------|----------|
| [`docs/index.md`](docs/index.md) | Índice geral da documentação |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Camadas, bootstrap e governança |
| [`docs/INFRASTRUCTURE.md`](docs/INFRASTRUCTURE.md) | Perfis, variáveis e Docker |
| [`docs/RELIABILITY.md`](docs/RELIABILITY.md) | Retries, telemetria e operação |
| [`docs/QUALITY_SCORE.md`](docs/QUALITY_SCORE.md) | Qualidade por área |
| [`docs/production-roadmap.md`](docs/production-roadmap.md) | Roadmap de produção |
| [`docs/product-specs/`](docs/product-specs/) | Especificações de produto |

---

*Calosum — framework de agente neuro-simbólico com hemisférios acoplados por contratos estáveis e governança mecânica.*
O compose agora inclui `healthcheck` chamando `/ready`. Os smokes versionados gerados neste plano estão em `docs/benchmarks/ci/2026-04-03-docker-profile-ready.*` e `docs/benchmarks/ci/2026-04-03-reflection-branching-smoke.*`.
