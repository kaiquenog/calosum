# Calosum Dual-Hemisphere Gap Closure Execution Plan

## Purpose

Fechar o delta entre a implementacao aspiracional ja aplicada no core e os gaps restantes que ainda impedem considerar a arquitetura dual-hemisferio operacionalmente fechada para uso local confiavel e para claims cientificos honestos.

## Scope

- Completar os itens remanescentes do plano aspiracional anterior que ainda nao foram entregues:
  - contrato/versionamento do backend `jepa-rs`
  - evolucao dedicada de `vljepa` multimodal
  - budget de CPU/memoria por backend
  - smoke docker e2e versionado
  - promocao controlada do branching multi-candidato de `opt-in` para contrato estavel
- Endurecer a camada de operacao para que readiness, benchmark e docker reflitam o estado real do sistema.
- Cobrir em testes e docs os novos contratos de reflection, bridge bidirecional e budget operacional.

Fora de escopo neste plano:

- Treino foundation-scale
- migracao cloud-first
- refatoracao ampla do dominio sem relacao com os gaps acima

## Validation

Validacao obrigatoria por sprint:

1. Governanca
   - `PYTHONPATH=src ./.venv/bin/python3 -m calosum.harness_checks`
2. Regressao
   - `PYTHONPATH=src ./.venv/bin/python3 -m unittest discover -s tests -t .`
3. Operacao
   - `GET /ready`
   - benchmark smoke com artefato versionado
4. Infra
   - `docker compose -f deploy/docker-compose.yml up -d`
   - smoke e2e do perfil `docker`

Critério de conclusão deste plano:

- nenhum backend declarado como cientifico sem contrato/spec/telemetria correspondente
- budgets por backend expostos e cobertos por teste
- docker profile pronto com smoke versionado
- branching multi-candidato com contrato estavel e benchmark explicito

## Progress

- [ ] Sprint 1: contratos honestos do hemisferio direito
- [ ] Sprint 2: hardening operacional e budgets
- [ ] Sprint 3: promocao controlada do branching e benchmark de reflection
- [ ] Sprint 4: docker e fechamento de documentacao

## Decision Log

- 2026-04-03: o plano anterior sai de `active/` porque a parte executavel local principal ja foi aplicada e validada.
- 2026-04-03: o novo plano foca somente o remanescente real, evitando manter em aberto entregas ja executadas.
- 2026-04-03: branching multi-candidato continua como mecanismo existente, mas a sua promocao para contrato default exige uma sprint propria para nao quebrar compatibilidade legada.

## Sprint 1

### Objetivo

Fechar os contratos e claims cientificos ainda incompletos do hemisferio direito.

### Entregas

1. Especificar formalmente o contrato `jepa-rs`.
2. Adicionar o artefato versionado de spec para o binario Rust/Burn.
3. Completar a evolucao multimodal de `vljepa` com fusao texto+visao antes da hierarquia.
4. Revisar a telemetria para refletir claramente:
   - checkpoint real presente/ausente
   - modo multimodal ativo/inativo
   - degradacao heuristica

### Mudancas de codigo

- `src/calosum/adapters/hemisphere/input_perception_vljepa.py`
- `src/calosum/adapters/hemisphere/input_perception_jepars.py`
- `docs/` ou `src/calosum/adapters/hemisphere/` com spec formal do contrato `jepa-rs`

### Testes necessarios

- `tests/adapters/hemisphere/test_vljepa_multimodal_fusion.py`
- `tests/adapters/hemisphere/test_jepars_contract.py`

## Sprint 2

### Objetivo

Endurecer o sistema para operacao local previsivel, com budgets explicitos por backend e readiness verificavel.

### Entregas

1. Expor budget de CPU/memoria por backend no bootstrap/introspection.
2. Adicionar gates/testes para readiness detalhado por componente.
3. Definir e testar fallback operacional quando o budget for excedido.
4. Registrar budgets e degradacoes no payload de readiness e telemetria.

### Mudancas de codigo

- `src/calosum/bootstrap/wiring/factory.py`
- `src/calosum/bootstrap/entry/api.py`
- `src/calosum/domain/metacognition/introspection_capabilities.py`
- `src/calosum/domain/infrastructure/telemetry.py`

### Testes necessarios

- `tests/bootstrap/test_readiness_component_health.py`
- testes de budget por backend em `tests/bootstrap/` ou `tests/domain/`

## Sprint 3

### Objetivo

Promover o branching multi-candidato de mecanismo experimental opt-in para contrato estavel, com benchmark e caminho de compatibilidade explicito.

### Entregas

1. Definir a estrategia de compatibilidade para `AgentTurnResult` vs `GroupTurnResult`.
2. Tornar o branching benchmarkavel de forma explicita.
3. Medir o impacto de `candidate_count >= 2` em latencia, `tool_success_rate` e retries.
4. Decidir se o default sera:
   - continuar `single-candidate`
   - ativar branching por perfil
   - ativar branching por heuristica com contrato unificado

### Mudancas de codigo

- `src/calosum/domain/agent/orchestrator.py`
- `src/calosum/shared/models/types.py`
- `scripts/ci_integration_benchmark.py`
- `docs/benchmarks/`

### Testes necessarios

- `tests/integration/test_group_turn_branching.py`
- `tests/domain/metacognition/test_reflection_multi_candidate.py`
- benchmark/smoke cobrindo `candidate_count >= 2`

## Sprint 4

### Objetivo

Fechar docker, compose e documentacao de execucao para encerrar o plano sem drift.

### Entregas

1. Smoke docker e2e versionado com JEPA e RLM configurados.
2. Teste `docker profile ready`.
3. Atualizar docs de arquitetura/infra/readiness para o estado final.
4. Encerrar o plano com criterios de saida auditaveis.

### Mudancas de codigo

- `deploy/docker-compose.yml`
- `docs/ARCHITECTURE.md`
- `docs/INFRASTRUCTURE.md`
- `docs/benchmarks/ci/`

### Testes necessarios

- `tests/integration/test_docker_profile_ready.py`
- qualquer smoke automatizado adicional sob `scripts/` ou `tests/integration/`

## Riscos

- Risco de promover defaults de branching sem contrato unificado e quebrar chamadas legadas.
- Risco de continuar usando nomes cientificos fortes em backends parcialmente heurísticos.
- Risco de docker continuar nominal se o smoke nao virar artefato reproduzivel.
- Risco de budgets existirem so em docs, sem enforcement/teste.
