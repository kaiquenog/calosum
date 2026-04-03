# Calosum Dual-Hemisphere Aspirational Implementation Plan

## Purpose

Transformar o Calosum do estado atual de framework neuro-simbolico parcialmente simulado para uma arquitetura dual-hemisferio coerente, testavel e operacionalmente viavel em 2026, preservando `Ports and Adapters`, `local-first` e governanca mecanica via `harness_checks.py`.

## Scope

- Corrigir gaps estruturais que hoje bloqueiam a credibilidade arquitetural:
  - docs desatualizadas
  - CI declarativa quebrada
  - configuracoes invalidas de deploy
  - defaults perigosos no bootstrap
- Tornar o hemisferio direito um world-model adapter real, com fronteira clara entre:
  - embeddings heuristicas
  - JEPA treinado local
  - backend `jepa-rs`
- Tornar o hemisferio esquerdo um planner recursivo real, removendo a simulacao AST baseada em `split(". ")`.
- Substituir o pseudo-GEA atual por branching, scoring e reflexao realmente multi-candidato.
- Introduzir bridge bidirecional baseado em sinais estruturados, nao em parsing de strings.
- Consolidar Active Inference, free energy e neuroplasticidade em modulos matematicos e loops operacionais separados.
- Endurecer seguranca e operacao do runtime/API para perfil local e docker.

Fora de escopo neste plano:

- Treino foundation-scale do JEPA
- Fine-tuning pesado de LLM grande dentro do repositorio principal
- Orquestracao cloud-first

## Validation

Validacao obrigatoria por etapa:

1. Governanca
   - `PYTHONPATH=src ./.venv/bin/python3 -m calosum.harness_checks`
2. Regressao
   - `PYTHONPATH=src ./.venv/bin/python3 -m unittest discover -s tests -t .`
3. Smokes por subsistema
   - hemisferio direito: testes dedicados para `heuristic_jepa`, `trained_jepa`, `vjepa21`, `vljepa`, `jepars`
   - hemisferio esquerdo: testes de recursao, reparo, custo e profundidade
   - bridge: testes de fusao, feedback estruturado e persistencia
   - reflection: testes de branching com `candidate_count > 1`
4. Operacao
   - smoke do `docker compose -f deploy/docker-compose.yml up -d`
   - `GET /ready`
   - um turno e2e em `persistent` e um em `docker`
5. Benchmark minimo
   - latencia p95 por turno
   - `tool_success_rate`
   - `runtime_retry_count`
   - taxa de fallback por backend

Ultima validacao registrada neste plano em `2026-04-03`:

- `PYTHONPATH=src ./.venv/bin/python3 -m calosum.harness_checks` -> `passed`
- `PYTHONPATH=src ./.venv/bin/python3 -m unittest discover -s tests -t .` -> `passed` (`Ran 198 tests`)
- smokes dedicados adicionados para:
  - branching multi-candidato
  - mismatch signal estruturado
  - latent exchange afetando selecao
  - multiagente com workflow real
  - OTLP com backoff
  - readiness detalhado
- ainda nao executado nesta rodada:
  - `docker compose -f deploy/docker-compose.yml up -d`
  - smoke e2e de perfil `docker`
  - budget explicito de CPU/memoria por backend

## Progress

- [x] Sprint 1: corrigir credibilidade operacional e remover inconsistencias fatais
- [ ] Sprint 2: tornar hemisferios e bridge mecanismos reais
- [ ] Sprint 3: tornar reflexao, neuroplasticidade e multiagente mecanismos reais
- [ ] Sprint 4: endurecer benchmark, observabilidade e readiness para producao local

Resumo final para arquivamento:

- concluido nesta linha de execucao:
  - Sprint 1 completo
  - parte executavel central de Sprint 2, Sprint 3 e Sprint 4
- validado:
  - `harness_checks`
  - suite unitaria/integracao local (`Ran 198 tests`)
- remanescente movido para novo plano ativo:
  - contrato/spec `jepa-rs`
  - evolucao multimodal dedicada de `vljepa`
  - budget de CPU/memoria por backend
  - smoke docker e2e
  - promocao controlada do branching para contrato estavel/default

Status atual em `2026-04-03`:

- Sprint 1: concluido.
- Sprint 2: parcialmente concluido. Telemetria honesta de backend, modulos matematicos e recursao real do RLM foram implementados; ainda faltam `right_hemisphere_vjepa2.py` e a spec formal separada do contrato `jepa-rs`.
- Sprint 3: parcialmente concluido. Branching real opt-in, scoring multi-candidato, learned selector no resolver, `StructuredMismatchSignal`, latent exchange no scoring e workflow multiagente real foram implementados.
- Sprint 4: parcialmente concluido. Benchmark smoke real em `scripts/`, OTLP com backoff/circuito simples e readiness detalhado por componente foram implementados; ainda faltam budget de CPU/memoria por backend e docker e2e completo.

Resumo executivo:

- Implementado no core:
  - bootstrap fail-closed para endpoint esquerdo
  - benchmark de CI real e gate funcional
  - readiness detalhado
  - reflection multi-candidato com telemetria
  - `LearnedPreferenceGEAReflectionController` no resolver via flag
  - bridge bidirecional com `StructuredMismatchSignal`
  - `latent_exchange` afetando selecao
  - recursao do hemisferio esquerdo por operacoes (`decompose`, `inspect`, `synthesize`, `verify`)
  - workflow multiagente real para subordinacao
  - OTLP com backoff
- Ainda pendente para fechamento total do plano:
  - budget operacional por backend
  - smoke docker e2e versionado
  - novos artefatos/specs externos prometidos explicitamente no Sprint 2

## Decision Log

- 2026-04-03: priorizar estabilizacao antes de expansao aspiracional. O repositrio hoje passa testes, mas ainda mistura mecanismos reais com simulacoes nominais.
- 2026-04-03: manter `Ports and Adapters`; novos backends entram apenas atras de `Protocol`.
- 2026-04-03: separar claramente "fallback heuristico" de "backend cientifico". Nomes e telemetria devem dizer a verdade.
- 2026-04-03: qualquer claim de GEA, RLM, V-JEPA ou Active Inference so permanece no core se houver comportamento observavel correspondente.
- 2026-04-03: branching multi-candidato foi implementado como mecanismo real, mas mantido `opt-in` por `CALOSUM_GEA_MAX_CANDIDATES` para preservar compatibilidade com o contrato legado de `AgentTurnResult`.
- 2026-04-03: readiness passou a expor health/componentes, enquanto docker e budget por backend permanecem como fechamento operacional pendente.
- 2026-04-03: o plano continua ativo porque a parte executavel local foi aplicada, mas ainda existem entregas explicitamente previstas que dependem de spec/artefato externo ou rodada dedicada de infra docker.

## Sprint 1

### Objetivo

Eliminar drift entre codigo, docs, deploy e CI. Sem isso, qualquer evolucao cognitiva continua assentada em base falsa.

### Entregas

1. Corrigir `docs/ARCHITECTURE.md` para refletir arquivos reais.
2. Corrigir workflow de CI para usar scripts existentes ou adicionar os scripts faltantes.
3. Corrigir `deploy/docker-compose.yml` para usar `CALOSUM_RIGHT_MODEL_PATH`.
4. Impedir autochamada recursiva do hemisferio esquerdo quando `CALOSUM_MODE=api` nao define endpoint.
5. Endurecer CORS e politicas default da API.
6. Criar benchmark scripts reais versionados sob `scripts/`.

### Mudancas de codigo

- `src/calosum/bootstrap/wiring/backend_resolvers.py`
  - falhar cedo se `mode=api` e `left_hemisphere_endpoint` nao estiver configurado
  - remover fallback silencioso para `QwenLeftHemisphereAdapter()` apontando para `localhost:8000`
- `src/calosum/adapters/llm/llm_qwen.py`
  - remover `api_url` default auto-referencial
  - exigir endpoint explicito ou provider local valido
- `deploy/docker-compose.yml`
  - trocar `CALOSUM_JEPA_MODEL_PATH` por `CALOSUM_RIGHT_MODEL_PATH`
- `.github/workflows/ci.yml`
  - alinhar com scripts reais
- `docs/ARCHITECTURE.md`
  - alinhar nomes e paths reais
- `src/calosum/bootstrap/entry/api.py`
  - restringir `allow_origins`
  - desabilitar `allow_credentials=True` com `*`

### Novas env vars

- `CALOSUM_ALLOWED_ORIGINS`
- `CALOSUM_REQUIRE_LEFT_ENDPOINT`
- `CALOSUM_CI_BENCHMARK_TURNS`

### Testes necessarios

- `tests/bootstrap/test_factory_requires_left_endpoint_in_api_mode.py`
- `tests/bootstrap/test_docker_env_alignment.py`
- `tests/bootstrap/test_api_cors_policy.py`
- `tests/test_ci_scripts_exist.py`

### Exemplo de direcao de implementacao

```python
# backend_resolvers.py
if settings.mode == CalosumMode.API and not settings.left_hemisphere_endpoint:
    raise RuntimeError(
        "CALOSUM_MODE=api requires CALOSUM_LEFT_ENDPOINT; refusing self-referential default."
    )
```

## Sprint 2

### Objetivo

Fazer os hemisferios deixarem de ser placeholders sofisticados.

### Entregas

1. Introduzir `RightHemisphereBackendDescriptor` com telemetria honesta:
   - `heuristic_literal`
   - `predictive_checkpoint`
   - `vjepa21_local`
   - `vljepa_local`
   - `jepars_local`
2. Extrair math de Active Inference para modulo dedicado:
   - `shared/utils/free_energy.py`
   - `shared/utils/surprise_metrics.py`
3. Reescrever `left_hemisphere_rlm_ast.py` para recursao real guiada por operacoes:
   - `decompose`
   - `inspect`
   - `synthesize`
   - `verify`
4. Tornar `vljepa` multimodal de fato:
   - fusao texto+visao antes de hierarquia
   - nao apenas analise do vetor ja pronto
5. Adicionar adapter Rust/Burn com contrato formal para `jepa-rs`.

### Status atual

- Implementado:
  - nomes honestos de backend (`heuristic_literal`, `predictive_checkpoint`, `vjepa21_local`, `vljepa_local`, `jepars_local`)
  - `shared/utils/free_energy.py`
  - `shared/utils/surprise_metrics.py`
  - recursao do `left_hemisphere_rlm_ast.py` por operacoes
- Parcial:
  - `vljepa` segue dependente do adapter existente; o plano aspiracional de fusao multimodal dedicada nao foi expandido nesta rodada
- Pendente:
  - `src/calosum/adapters/hemisphere/right_hemisphere_vjepa2.py`
  - spec de contrato isolada para `jepa-rs`

### Novos adapters

- `src/calosum/adapters/hemisphere/right_hemisphere_vjepa2.py`
- `src/calosum/adapters/hemisphere/right_hemisphere_jepars.rs.md` (spec de contrato do binario)
- `src/calosum/adapters/hemisphere/left_hemisphere_rlm_runtime.py`
- `src/calosum/adapters/hemisphere/left_hemisphere_rlm_ast.py` (reduzido a fallback ou removido)

### Funcoes matematicas

- `expected_free_energy_refined(...)`
  - separar valor instrumental e valor epistemico
  - adicionar weighting por novelty e reliability
- `variational_free_energy(...)`
  - KL + reconstruction/prediction error + uncertainty regularizer
- `surprise_from_predictive_error(...)`
  - evitar logistic overflow
  - saturacao numericamente estavel
- `hierarchical_latent_prediction(...)`
  - predicao coarse-to-fine para `vljepa`

### Testes necessarios

- `tests/adapters/hemisphere/test_vjepa2_contract.py`
- `tests/adapters/hemisphere/test_vljepa_multimodal_fusion.py`
- `tests/adapters/hemisphere/test_jepars_contract.py`
- `tests/adapters/hemisphere/test_rlm_recursive_runtime.py`
- `tests/adapters/perception/test_free_energy_math.py`

### Exemplo de direcao de implementacao

```python
@dataclass(slots=True)
class FreeEnergyTerms:
    epistemic_value: float
    instrumental_value: float
    complexity: float
    novelty_bonus: float

def expected_free_energy_refined(...)-> tuple[float, FreeEnergyTerms]:
    ...
```

## Sprint 3

### Objetivo

Substituir o "GEA de candidato unico" por branching real, reflexao real e bridge bidirecional estrutural.

### Entregas

1. `CalosumAgent.aprocess_turn()` deve gerar N candidatos quando:
   - surpresa alta
   - incerteza alta
   - repair loop recorrente
   - tarefa explicitamente complexa
2. `GEAReflectionController` deve selecionar entre multiplos candidatos reais.
3. `LearnedPreferenceGEAReflectionController` deve entrar no resolver quando habilitado.
4. O bridge bidirecional deve receber um `StructuredMismatchSignal` vindo do verificador/runtime, em vez de procurar substrings em `reasoning_summary`.
5. `latent_exchange` deve ser usado no scoring ou removido.
6. Multiagente deve parar de ser simulacao fixa e virar orquestracao de workers reais ou ser rebaixado de escopo.

### Status atual

- Implementado:
  - branching real em `CalosumAgent.aprocess_turn()`
  - reflexao multi-candidato com scoring por EFE + qualidade + sinal social
  - learned selector habilitavel no resolver
  - `StructuredMismatchSignal`
  - `latent_exchange` no scoring
  - workflow multiagente real para a tool de subordinacao
- Decisao operacional:
  - o mecanismo existe e tem testes dedicados, mas o default continua single-candidate para nao quebrar chamadas legadas que assumem `AgentTurnResult`

### Mudancas de codigo

- `src/calosum/domain/agent/orchestrator.py`
- `src/calosum/domain/execution/agent_execution.py`
- `src/calosum/domain/metacognition/metacognition.py`
- `src/calosum/bootstrap/wiring/backend_resolvers.py`
- `src/calosum/adapters/experience/gea_reflection_experience.py`
- `src/calosum/adapters/communication/latent_exchange.py`

### Novas env vars

- `CALOSUM_GEA_MAX_CANDIDATES`
- `CALOSUM_GEA_ENABLE_LEARNED_SELECTOR`
- `CALOSUM_GEA_BRANCH_SURPRISE_THRESHOLD`
- `CALOSUM_GEA_BRANCH_COMPLEXITY_THRESHOLD`

### Testes necessarios

- `tests/integration/test_group_turn_branching.py`
- `tests/domain/metacognition/test_reflection_multi_candidate.py`
- `tests/domain/cognition/test_structured_mismatch_signal.py`
- `tests/integration/test_latent_exchange_affects_selection.py`

### Exemplo de direcao de implementacao

```python
candidates = await self.execution_engine.run_candidates(
    user_turn=user_turn,
    variants=self._build_variants(right_state, workspace),
)
outcome = await self.reflection_controller.aevaluate(candidates, self.tokenizer)
selected = _select_candidate(candidates, outcome.selected_variant_id)
```

## Sprint 4

### Objetivo

Fechar o sistema para operacao local confiavel.

### Entregas

1. Benchmarks reais em `scripts/` com artefatos versionados.
2. OTLP endurecido com retry/backoff e circuito aberto.
3. Budget de CPU/memoria por backend.
4. Health/readiness detalhados por componente.
5. Teste docker e2e com JEPA e RLM configurados.

### Status atual

- Implementado:
  - benchmark smoke real em `scripts/`
  - OTLP com retry/backoff simples e circuito via janela de supressao
  - readiness detalhado por componente
- Pendente:
  - budget de CPU/memoria por backend
  - teste docker e2e versionado

### Testes necessarios

- `tests/adapters/infrastructure/test_telemetry_otlp_backoff.py`
- `tests/bootstrap/test_readiness_component_health.py`
- `tests/integration/test_docker_profile_ready.py`

### Criterios de saida

- CI reproduzivel sem referencias quebradas
- nenhum adapter critico dependendo de heuristica silenciosa sem telemetria explicita
- reflection com `candidate_count >= 2` coberto por teste e benchmark
- API sem defaults perigosos
- docker compose sobe com configuracao coerente

Estado atual dos criterios:

- atendido: CI/docs/defaults/telemetria/branching testado
- atendido parcialmente: reflection com `candidate_count >= 2` coberto por teste, mas ainda nao promovido a benchmark/gate default
- pendente: validacao docker compose + e2e docker

## Sequenciamento Executivo

1. Estabilizar CI/docs/deploy/defaults.
2. Reescrever matematica e contratos do hemisferio direito.
3. Reescrever recursao do hemisferio esquerdo.
4. Introduzir branching e reflexao real.
5. Endurecer operacao e benchmark.

## Riscos

- Risco de continuar adicionando nomes SOTA sobre mecanismos heuristicos.
- Risco de custo computacional inviavel se `vjepa` e `rlm` forem acoplados sem budget.
- Risco de regressao de confiabilidade se branching entrar antes de budgets e observabilidade.
- Risco de seguranca se tools de rede/execucao crescerem sem allowlists e auditoria real.

## Debt a registrar se aparecer durante a execucao

- Todo fallback heuristico que permanecer em caminho principal
- Todo adapter que declare backend cientifico sem checkpoint ou contrato real
- Toda regra arquitetural recorrente ainda nao mecanizada no harness
