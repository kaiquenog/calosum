# Relatório de Análise de Maturidade — Calosum

**Data:** 2026-03-29
**Versão:** 0.1.0
**Analista:** Claude Code (Opus 4.6)

---

## Sumário Executivo

O Calosum é um framework neuro-simbólico de agente cognitivo com arquitetura dual-hemisférica inspirada na neurociência. O projeto demonstra **maturidade arquitetural avançada** para um projeto em fase alpha, com separação de concerns exemplar, governança via AST enforcement, e um pipeline cognitivo completo (percepção → raciocínio → execução → reflexão → neuroplasticidade).

**Veredicto geral: B- (Fundação Sólida, Pré-Produção)**

O projeto possui a estrutura de um sistema production-grade, mas ainda opera com adaptadores parciais, sem CI/CD automatizado, e com gaps significativos em observabilidade de produção e internacionalização.

---

## 1. Métricas Quantitativas

### 1.1 Dimensão do Código

| Camada | Arquivos | LOC (aprox.) | % do Total |
|--------|----------|-------------|------------|
| `shared/` (Contratos) | 5 | ~523 | 8% |
| `domain/` (Lógica core) | 15 | ~3.651 | 55% |
| `adapters/` (Implementações) | 8 | ~1.591 | 24% |
| `bootstrap/` (Entrada/DI) | 5 | ~799 | 12% |
| `harness_checks.py` | 1 | ~368 | 6% |
| **Total Python (src/)** | **34** | **~6.932** | **100%** |
| Testes | 15 | ~1.200 | — |
| UI (React/TS) | 4 | ~1.068 | — |
| Documentação | 28+ | N/A | — |

### 1.2 Saúde dos Testes

| Métrica | Valor |
|---------|-------|
| Total de testes | 37 |
| Taxa de sucesso | 100% (37/37 OK) |
| Tempo de execução | 0.759s |
| Classes de teste | 14 |
| Assertions totais | ~130+ |
| Cobertura de camadas | domain, adapters, bootstrap, harness |
| Testes assíncronos | 6 classes (IsolatedAsyncioTestCase) |

### 1.3 Dependências

| Categoria | Contagem | Exemplos |
|-----------|----------|----------|
| Python (requirements.txt) | ~80 | torch, transformers, qdrant-client, fastapi, httpx |
| Node.js (package.json) | ~25 | React 19, Vite 8, Tailwind 4.2, TypeScript 5.9 |
| Serviços externos | 4 | OpenAI API, Qdrant, OTel Collector, Jaeger |

---

## 2. Análise de Maturidade por Dimensão

### 2.1 Arquitetura — Score: A-

**Pontos Fortes:**
- **Hexagonal Architecture (Ports & Adapters)** implementada com rigor: `shared/ports.py` define 10 Protocols, todos `@runtime_checkable`
- **Dependency Inversion** total: domain nunca importa de adapters/bootstrap
- **Governança mecânica** via `harness_checks.py`: 46 regras de módulo validadas por AST, limites de tamanho (<400 LOC), documentação obrigatória
- **Camadas bem definidas**: shared → domain → adapters → bootstrap (fluxo unidirecional)
- **Builder pattern** para injeção de dependências via `CalosumAgentBuilder`
- **Fallback gracioso**: cada componente opera mesmo sem infraestrutura completa (Qdrant → JSONL → in-memory)

**Pontos de Melhoria:**
- `advanced_interfaces.py` (126 LOC) é código morto — ABCs não utilizadas que duplicam os Protocols de `ports.py`
- Detecção de provider no factory usa heurísticas de hostname frágeis (hardcoded URL patterns)
- `left_hemisphere.py` viola parcialmente SRP: composição de resposta misturada com geração de ações

**Recomendações:**
1. Remover `advanced_interfaces.py` ou converter em documentação
2. Extrair detecção de provider para um módulo `adapters/provider_detection.py` com registry pattern
3. Separar `LeftHemisphereLogicalSLM` em `Reasoner` (raciocínio) e `ResponseComposer` (formatação)

---

### 2.2 Domain Model — Score: A-

**Pontos Fortes:**
- **Ubiquitous Language** consistente: UserTurn, CognitiveBridgePacket, SoftPromptToken, PrimitiveAction
- **Value Objects** imutáveis com `@dataclass(slots=True)` em todo lugar — eficiência de memória
- **20+ tipos de domínio** modelando o pipeline cognitivo completo
- **Active Inference** implementado: surprise/ambiguity triggers branching automático
- **Self-correction loop**: verifier → critique → repair → retry
- **Lambda DSL** com parser seguro (AST whitelisting, sem imports possíveis)

**Pontos de Melhoria:**
- Tipos sem validação (Pydantic usado apenas para settings, não para domain types)
- `CritiqueVerdict.confidence` hardcoded em 0.8 — deveria refletir severidade real
- Keywords de salience/emoção em português hardcoded no domain (deveria estar em config)
- `multiagent.py` é esquelético (59 LOC, apenas stubs)
- `event_bus.py` não cancela worker task no shutdown (potential leak)

**Recomendações:**
1. Adicionar validators nos dataclasses críticos (salience ∈ [0,1], temperature > 0)
2. Extrair keywords de i18n para arquivos de configuração carregáveis
3. Implementar shutdown gracioso no EventBus (`stop()` method com cancellation)
4. Evoluir multiagent.py de stubs para implementação real ou remover

---

### 2.3 Adapters & Integrações — Score: C+

**Pontos Fortes:**
- `llm_qwen.py`: Auto-detecção de 3 modos de API (Responses, Chat, Compatible)
- `text_embeddings.py`: 3 fallback tiers (OpenAI → HuggingFace → lexical determinístico)
- `action_runtime.py`: Sandboxing real (tempdir para write_file, DuckDuckGo para search)
- `memory_qdrant.py`: Reconstrução robusta de domain objects a partir de pontos vetoriais
- `night_trainer.py`: Pipeline de consolidação com dataset export

**Pontos de Melhoria:**
- **Sem retry/backoff** em chamadas HTTP (LLM, embeddings, Qdrant)
- **Sem connection pooling** configurável
- **Sem circuit breaker** — falha de um serviço cascata para todo o pipeline
- **Night trainer é mock DSPy**: lógica real de otimização não implementada
- **bridge_store.py** falha silenciosamente sem logging
- **Sem rate limiting** nas chamadas de API
- **HTTP timeout de 300s** sem configuração granular

**Recomendações:**
1. **[CRÍTICO]** Adicionar retry com exponential backoff via `tenacity` ou `httpx` retry transport
2. **[CRÍTICO]** Implementar circuit breaker pattern (ex: `pybreaker`) para Qdrant e LLM
3. Adicionar connection pooling configurável no httpx.AsyncClient
4. Implementar rate limiter para chamadas de API (token bucket)
5. Adicionar logging estruturado no `bridge_store.py`
6. Implementar DSPy real no `night_trainer.py` ou documentar como roadmap item

---

### 2.4 Testes — Score: B

**Pontos Fortes:**
- **37 testes, 100% passando** em <1 segundo
- **Cobertura comportamental** forte: testa o quê, não como
- **Mocks sofisticados**: FakeQdrantClient, FaultyLeftHemisphere com comportamento realista
- **Testes de persistência**: verificam sobrevivência de estado entre reloads
- **Testes de governança**: `test_harness.py` valida toda a arquitetura
- **6 classes async** com cleanup adequado

**Pontos de Melhoria:**
- **Sem CI/CD** — testes rodam apenas localmente
- **Sem coverage report** configurado (`.coverage` existe mas sem `.coveragerc`)
- **Sem pytest** — usa unittest stdlib (menos expressivo)
- **Sem parametrize** — variantes testadas manualmente
- **Sem testes de contrato** para APIs HTTP
- **Sem testes de stress/load** para memory consolidation
- **Sem testes de mutation** para verificar qualidade dos assertions

**Áreas Não Cobertas:**
| Componente | Status | Risco |
|-----------|--------|-------|
| Memory consolidation edge cases | Não testado | MÉDIO |
| Event bus async lifecycle | Não testado | MÉDIO |
| Bridge neural projection (PyTorch) | Não testado | MÉDIO |
| Branching multi-variant paralelo | Não testado | MÉDIO |
| API endpoints (FastAPI) | Não testado | ALTO |
| UI (React) | Não testado | BAIXO |
| Race conditions em async | Não testado | MÉDIO |

**Recomendações:**
1. **[CRÍTICO]** Configurar GitHub Actions com `unittest discover` + coverage report
2. Migrar para `pytest` para ganhar fixtures, parametrize, e melhor reporting
3. Adicionar testes de API com `httpx.AsyncClient` + `TestClient` do FastAPI
4. Configurar `.coveragerc` com meta de >80% e fail threshold
5. Adicionar testes de contrato para os Protocols (property-based com `hypothesis`)
6. Criar `tests/test_api.py` para endpoints REST e SSE

---

### 2.5 Observabilidade — Score: B-

**Pontos Fortes:**
- **5 canais de telemetria**: felt, thought, decision, execution, reflection
- **Trace ID + Span ID** por evento, compatível com OpenTelemetry
- **JSONL persistente** para audit trail
- **Dashboard React** com visualização em tempo real
- **Métricas por turno**: latência, retry count, rejection count
- **Docker Compose** com Jaeger + OTel Collector pré-configurados

**Pontos de Melhoria:**
- **Sem batching** de escrita — cada evento é uma operação de I/O
- **Sem alerting** — nenhum threshold configurado para paging
- **Sem métricas de histograma** (p50/p95/p99 latency)
- **Sem health check endpoint** no FastAPI
- **Sem structured logging** (usa `logging.warning` sem formato padronizado)
- **OTLP exporter direto** não implementado (ainda via JSONL)
- **Dashboard polling** a cada 2.5s — sem WebSocket push

**Recomendações:**
1. Adicionar `GET /health` e `GET /ready` no FastAPI
2. Implementar batched telemetry writer (buffer de 100 eventos ou 5s timeout)
3. Migrar logging para `structlog` com JSON output
4. Adicionar OTLP gRPC exporter direto (eliminar intermediário JSONL)
5. Evoluir dashboard para WebSocket para eliminar polling
6. Configurar alertas básicos (latency > 5s, rejection rate > 10%)

---

### 2.6 Segurança — Score: B

**Pontos Fortes:**
- **Lambda DSL sandboxed**: AST whitelist rejeita Import, Attribute, Subscript
- **Action whitelist**: StrictLambdaRuntime rejeita ações desconhecidas
- **External side effects bloqueados** por padrão
- **File write sandboxed** via tempdir
- **Permission system** com approval workflow
- **Verifier** detecta prompt injection keywords

**Pontos de Melhoria:**
- **CORS wildcard** (`allow_origins=["*"]`) — inseguro para produção
- **API sem autenticação** — qualquer um pode chamar endpoints
- **Vault via env vars** — sem rotação ou encryption at rest
- **Prompt injection detection** via substring matching — facilmente contornável
- **.env com API keys** no repositório (apenas gitignored)
- **Sem rate limiting** na API

**Recomendações:**
1. **[CRÍTICO]** Restringir CORS origins para domínios específicos em produção
2. **[CRÍTICO]** Adicionar autenticação na API (JWT ou API key middleware)
3. Implementar rate limiting (ex: `slowapi` ou `fastapi-limiter`)
4. Evoluir detecção de prompt injection para classificador ML
5. Adicionar secret management adequado (HashiCorp Vault, AWS Secrets Manager)
6. Implementar audit log para chamadas de API com write side effects

---

### 2.7 DevOps & CI/CD — Score: D

**Pontos Fortes:**
- Docker Compose funcional com 4 serviços
- Dockerfile clean (Python 3.13-slim, sem cache)
- `.gitignore` abrangente
- Harness checks como "CI local"

**Pontos de Melhoria:**
- **Zero CI/CD automatizado** — sem GitHub Actions, GitLab CI, ou similar
- **Sem linting automatizado** (ruff, mypy referenciados no .gitignore mas não configurados)
- **Sem pre-commit hooks**
- **Sem deploy automatizado**
- **Sem versionamento semântico** automatizado
- **Sem branch protection rules**
- **Sem environment promotion** (dev → staging → prod)

**Recomendações:**
1. **[CRÍTICO]** Criar `.github/workflows/ci.yml`:
   ```yaml
   - Run harness_checks
   - Run unittest discover
   - Run coverage report
   - Run ruff check + ruff format --check
   - Run mypy (strict mode)
   ```
2. Configurar pre-commit hooks (ruff, mypy, harness_checks)
3. Adicionar `[tool.ruff]` e `[tool.mypy]` no `pyproject.toml`
4. Configurar branch protection no GitHub (require CI pass, require review)
5. Implementar release workflow com semantic versioning

---

### 2.8 Documentação — Score: A-

**Pontos Fortes:**
- **28+ documentos** cobrindo arquitetura, infraestrutura, qualidade, planos
- **Execution Plans versionados** com template padronizado
- **20+ planos completados** — histórico de decisões preservado
- **Tech Debt Tracker** explícito
- **Design Docs** com rationale (core-beliefs, dspy-self-learning)
- **AGENTS.md** como guia de navegação rápida
- **Harness valida** presença de documentos obrigatórios

**Pontos de Melhoria:**
- **README em português** sem versão em inglês
- **Sem API docs auto-geradas** (FastAPI tem swagger, mas não configurado)
- **QUALITY_SCORE.md** desatualizado (2026-03-27, pré-últimas mudanças)
- **Sem changelog** ou release notes
- **Docs de design-docs/** parcialmente incompletos

**Recomendações:**
1. Habilitar Swagger UI no FastAPI (`docs_url="/docs"`)
2. Adicionar CHANGELOG.md com formato Keep a Changelog
3. Atualizar QUALITY_SCORE.md refletindo evolução recente
4. Adicionar README bilíngue (pt-BR + en)

---

### 2.9 UI / Frontend — Score: C+

**Pontos Fortes:**
- React 19 + Vite 8 + Tailwind 4 — stack moderno
- Dashboard com 5 canais cognitivos visualizados
- Chat SSE em tempo real
- Dark mode com CSS variables
- Session management com auto-discovery

**Pontos de Melhoria:**
- **Monolito**: 945 LOC em um único `App.tsx`
- **Zero testes frontend**
- **Sem componentização** — tudo inline no App
- **Sem state management** (useState para tudo)
- **Sem error boundaries** React
- **Sem acessibilidade** (ARIA labels, keyboard nav)
- **Sem responsividade** adequada (breakpoint único em 1024px)
- **Polling 2.5s** em vez de WebSocket

**Recomendações:**
1. Decompor `App.tsx` em componentes: `ChatPanel`, `TimelinePanel`, `SessionSelector`, `EventCard`
2. Adicionar React Error Boundaries
3. Implementar testes com Vitest + React Testing Library
4. Migrar state complexo para `useReducer` ou Zustand
5. Adicionar ARIA labels e keyboard navigation
6. Evoluir de polling para WebSocket

---

## 3. Matriz de Maturidade Consolidada

| Dimensão | Score | Nível | Descrição |
|----------|-------|-------|-----------|
| Arquitetura | A- | **Madura** | Hexagonal com governança mecânica |
| Domain Model | A- | **Madura** | Tipos ricos, pipeline cognitivo completo |
| Adapters/Integrações | C+ | **Parcial** | Funcional mas sem resiliência |
| Testes | B | **Boa** | Cobertura comportamental sólida, sem CI |
| Observabilidade | B- | **Adequada** | Telemetria completa, sem alerting |
| Segurança | B | **Adequada** | Sandboxing forte, API exposta |
| DevOps/CI-CD | D | **Imatura** | Zero automação |
| Documentação | A- | **Madura** | Rica e governada |
| Frontend | C+ | **Parcial** | Funcional mas monolítica |
| **MÉDIA PONDERADA** | **B-** | **Pré-Produção** | Fundação forte, gaps operacionais |

---

## 4. Roadmap de Melhoria Priorizado

### Sprint 1: Fundação Operacional (Impacto Alto, Esforço Baixo)

| # | Ação | Impacto | Esforço |
|---|------|---------|---------|
| 1 | Criar GitHub Actions CI (testes + harness + lint) | CRÍTICO | 2h |
| 2 | Configurar ruff + mypy no pyproject.toml | ALTO | 1h |
| 3 | Adicionar pre-commit hooks | ALTO | 30min |
| 4 | Habilitar Swagger UI no FastAPI | MÉDIO | 15min |
| 5 | Adicionar `/health` e `/ready` endpoints | MÉDIO | 30min |
| 6 | Restringir CORS em produção | CRÍTICO | 15min |

### Sprint 2: Resiliência (Impacto Alto, Esforço Médio)

| # | Ação | Impacto | Esforço |
|---|------|---------|---------|
| 7 | Retry + exponential backoff em httpx calls | CRÍTICO | 3h |
| 8 | Circuit breaker para Qdrant e LLM | ALTO | 2h |
| 9 | Autenticação na API (JWT/API key) | CRÍTICO | 4h |
| 10 | Rate limiting na API | ALTO | 2h |
| 11 | Structured logging com structlog | MÉDIO | 3h |
| 12 | Batched telemetry writer | MÉDIO | 2h |

### Sprint 3: Qualidade Expandida (Impacto Médio, Esforço Médio)

| # | Ação | Impacto | Esforço |
|---|------|---------|---------|
| 13 | Migrar para pytest + coverage >80% | ALTO | 4h |
| 14 | Testes de API (FastAPI TestClient) | ALTO | 3h |
| 15 | Testes de event bus lifecycle | MÉDIO | 2h |
| 16 | Testes de memory consolidation edges | MÉDIO | 2h |
| 17 | Componentizar App.tsx (4-5 componentes) | MÉDIO | 3h |
| 18 | Vitest + React Testing Library básico | MÉDIO | 2h |

### Sprint 4: Produção (Impacto Alto, Esforço Alto)

| # | Ação | Impacto | Esforço |
|---|------|---------|---------|
| 19 | OTLP gRPC exporter direto | ALTO | 4h |
| 20 | Secret management (Vault/AWS SM) | ALTO | 6h |
| 21 | WebSocket no dashboard (substituir polling) | MÉDIO | 4h |
| 22 | i18n extraction (keywords → config) | MÉDIO | 3h |
| 23 | DSPy real no night_trainer | ALTO | 8h |
| 24 | Branch protection + semantic release | MÉDIO | 2h |

---

## 5. Riscos Identificados

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| API key leak via .env | MÉDIA | ALTO | Secret manager + rotação |
| Cascata de falha sem circuit breaker | ALTA | ALTO | Implementar pybreaker |
| Regressão sem CI | ALTA | MÉDIO | GitHub Actions urgente |
| Data loss em persistent_memory sem schema versioning | BAIXA | ALTO | Adicionar versão no JSONL header |
| Event bus leak em shutdown | MÉDIA | BAIXO | Implementar graceful shutdown |
| Prompt injection contornando substring match | MÉDIA | MÉDIO | Evoluir para classificador ML |
| Monolito React incontrolável | MÉDIA | BAIXO | Componentizar antes de crescer |

---

## 6. Destaques Positivos (O que Está Muito Bem Feito)

1. **Governança Arquitetural Mecânica**: O `harness_checks.py` com 46 regras de importação validadas por AST é excepcional para um projeto deste tamanho. Impede drift arquitetural antes que aconteça.

2. **Pipeline Cognitivo Completo**: O fluxo percepção → tokenização → raciocínio → execução → reflexão → neuroplasticidade é coerente e bem modelado. A maioria dos projetos de agente para na execução.

3. **Active Inference**: O branching automático baseado em surprise score é uma implementação sofisticada de free energy principle que vai além de agent frameworks convencionais.

4. **Lambda DSL Seguro**: O parser com AST whitelisting que rejeita nodes perigosos é uma abordagem de segurança robusta para execução de programas gerados por LLM.

5. **Fallback Cascade**: O design onde cada componente degrada graciosamente (Qdrant → JSONL → RAM, HuggingFace → JEPA heurístico, OpenAI → lexical) garante que o sistema funciona em qualquer ambiente.

6. **Protocol-Based DI**: Usar `typing.Protocol` com `@runtime_checkable` em vez de ABCs permite duck typing seguro sem acoplamento de herança.

7. **Execution Plans Versionados**: O sistema de planejamento com 20+ planos completados documenta não apenas o quê foi feito, mas por quê — memória institucional valiosa.

8. **Testes em <1 segundo**: 37 testes em 0.76s é excelente. O fato de não depender de serviços externos para testes permite iteração rápida.

---

## 7. Conclusão

O Calosum é um projeto com **visão arquitetural de nível sênior** e **execução disciplinada**. A fundação é sólida o suficiente para suportar crescimento significativo, mas os gaps operacionais (CI/CD, resiliência de rede, segurança de API) precisam ser endereçados antes de qualquer uso em produção.

O maior risco imediato é a **ausência de CI/CD**, que significa que toda a governança arquitetural (que é excelente) depende de execução manual. O segundo risco é a **falta de resiliência em chamadas de rede**, que pode causar falhas cascateadas em ambiente real.

As recomendações estão ordenadas para maximizar impacto com mínimo esforço. Os Sprints 1 e 2 sozinhos elevariam o projeto de **B-** para **B+**, fechando os gaps operacionais mais críticos.

---

*Relatório gerado automaticamente via análise estática e dinâmica do código-fonte, testes, documentação e infraestrutura do projeto Calosum.*
