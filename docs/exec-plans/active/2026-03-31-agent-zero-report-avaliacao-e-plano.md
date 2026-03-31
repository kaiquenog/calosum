# 2026-03-31 - Avaliacao do Report Agent Zero e Plano de Execucao

## Purpose

Avaliar o report `agent_zero_analysis_report.md.resolved` contra o estado real do Calosum e sua documentacao oficial (`docs/ARCHITECTURE.md`, `docs/PLANS.md`, `docs/QUALITY_SCORE.md`, `docs/RELIABILITY.md`, `docs/INFRASTRUCTURE.md`, `docs/production-roadmap.md`), e transformar somente os pontos aplicaveis em um plano executavel por sprints.

## Scope

### Avaliacao dos pontos citados no report

1. Sessao de acao persistente (stateful tools): **parcialmente aplicavel**
- Estado atual: ja existe `PersistentShellTool` e wiring em `ConcreteActionRuntime` para `execute_bash` com sessao por `session_id`.
- Gap real: hardening (stderr/exit code/timeout/lifecycle), governanca de contrato de tool, testes de regressao.

2. SKILL.md e MCP: **aplicavel com escopo controlado**
- Estado atual: nao existe integracao MCP nativa no runtime do Calosum.
- Direcao: integrar MCP via Ports and Adapters (sem quebrar contratos atuais), e tratar SKILL.md como opcional de interoperabilidade, nao como substituicao abrupta das regras atuais.

3. Hierarquia de subagentes (recursao de contexto): **aplicavel com restricoes**
- Estado atual: existe base `domain/multiagent.py`, mas sem acao produtiva no loop principal para spawn delegado controlado.
- Direcao: adicionar delegacao com budget, profundidade maxima e isolamento.

4. Prompt-as-data (prompts fora de classes Python): **aplicavel**
- Estado atual: prompt base do hemisferio esquerdo ainda esta em codigo (`adapters/llm_payloads.py`), com artefatos noturnos complementares.
- Direcao: externalizar prompt base para arquivos versionados em diretorio dedicado, com loader e fallback.

5. Sistema de extensoes/hooks: **parcialmente aplicavel**
- Estado atual: existe `InternalEventBus` no orquestrador, que cobre parte do problema de extensao.
- Gap real: contrato formal de interceptacao/plugin para pontos estaveis de ciclo cognitivo.

### Fora de escopo neste plano

- Reescrever a arquitetura atual fora do padrao Ports and Adapters.
- Introduzir dependencias obrigatorias que removam fallbacks graciosos ja descritos em `docs/INFRASTRUCTURE.md`.

### Sprints planejadas

### Sprint 0 - Baseline, contrato e governanca

Objetivo:
- Congelar baseline tecnico e definir contratos minimos antes de mudancas cross-cutting.

Entregas:
- Documento de decisao curta com limites de escopo para MCP, subagentes e hooks.
- Mapa de impactos por camada (`shared/domain/adapters/bootstrap`) e registro dos novos modulos em `MODULE_RULES`.
- Checklist de compatibilidade com `docs/ARCHITECTURE.md` e `docs/RELIABILITY.md`.

Criterios de aceite:
- Nenhuma quebra de fronteira arquitetural no desenho.
- Plano de extracao para qualquer modulo em risco de 400 linhas tocado pelas sprints.

### Sprint 1 - Hardening de runtime persistente

Objetivo:
- Consolidar o estado persistente de shell como capability confiavel de producao.

Entregas:
- Ajustes no runtime para retorno estruturado (stdout/stderr/exit_code/duration).
- Controle de timeout e encerramento explicito de sessoes.
- Alinhamento de contrato de nome/descricao de tool (`execute_bash` vs persistencia interna).
- Telemetria de sessao e falhas de comando.

Testes da sprint:
- Testes unitarios para persistencia de estado (`cd`, variaveis de ambiente, arquivos temporarios por sessao).
- Testes de timeout e recuperacao apos processo morto.

Criterios de aceite:
- Fluxo multi-comando em uma mesma sessao funciona sem reinicializacao.
- Sem regressao no loop de repair do `AgentExecutionEngine`.

### Sprint 2 - Fundacao MCP (adapter e runtime tool)

Objetivo:
- Habilitar consumo de servidores MCP externos sem acoplamento no dominio.

Entregas:
- Novo adapter MCP atras de port/protocol dedicado.
- Tool de runtime para chamadas MCP com allowlist de metodos/servidores.
- Configuracoes de bootstrap/settings para habilitar/desabilitar MCP por ambiente.
- Documentacao de operacao e fallback quando MCP estiver indisponivel.

Testes da sprint:
- Teste com servidor MCP fake para sucesso, timeout e erro de contrato.
- Testes de permissao/politica para bloqueio de servidor nao permitido.

Criterios de aceite:
- MCP funciona como extensao opcional, sem impacto no fluxo default quando desativado.
- Erros de MCP nao derrubam o turno; apenas degradam a acao com telemetria clara.

### Sprint 3 - Delegacao por subagente controlada

Objetivo:
- Permitir delegacao de subtarefas sem poluir contexto principal e sem recursao infinita.

Entregas:
- Nova acao de delegacao (ex.: `spawn_subordinate`) com limites de profundidade, tempo e custo.
- Estrategia de agregacao de resultado de subagente no agente pai.
- Integracao com `domain/multiagent.py` e event bus para rastreabilidade.

Testes da sprint:
- Testes de budget/depth limit.
- Testes de retorno de resultado e propagacao de erro de subagente.

Criterios de aceite:
- Delegacao opcional e segura, sem quebrar o fluxo de turno simples/group turn existente.

### Sprint 4 - Prompt-as-data e contrato de hooks/interceptors

Objetivo:
- Separar prompt base de codigo e formalizar pontos de extensao estaveis.

Entregas:
- Diretorio versionado de prompts base (ex.: `prompts/left_hemisphere/`).
- Loader de prompt com fallback para defaults seguros.
- Contrato de interceptor/hook (antes/depois de percepcao, raciocinio, runtime e verificacao).
- Adaptacao do orquestrador para chamar interceptores sem acoplamento forte.

Testes da sprint:
- Testes de carga de prompt por arquivo e fallback.
- Testes de ordem de execucao e isolamento de erros dos hooks.

Criterios de aceite:
- Ajuste de prompt sem edicao de codigo core.
- Hooks nao comprometem disponibilidade do turno (degradacao controlada).

### Sprint 5 - Criacao/execucao de testes e verificacao de harness (gate final obrigatorio)

Objetivo:
- Fechar com evidencias de qualidade e governanca do repositorio.

Entregas:
- Criacao/atualizacao de testes unitarios e de integracao cobrindo todas as sprints aplicadas.
- Execucao obrigatoria de harness checks:
  - `PYTHONPATH=src python3 -m calosum.harness_checks`
- Execucao obrigatoria da suite de testes:
  - `PYTHONPATH=src python3 -m unittest discover -s tests -t .`
- Registro de resultados e eventuais debitos remanescentes em `docs/exec-plans/tech-debt-tracker.md`.

Criterios de aceite:
- Harness verde.
- Suite de testes verde.
- Nenhuma violacao de fronteira arquitetural introduzida.

## Validation

- Validar toda sprint contra:
  - `docs/ARCHITECTURE.md` (fronteiras e composicao)
  - `docs/PLANS.md` (estrutura de plano e governanca)
  - `docs/RELIABILITY.md` (retries, fallback, observabilidade)
  - `docs/QUALITY_SCORE.md` (gaps e criterio de elevacao de score)
  - `docs/INFRASTRUCTURE.md` (profiles/env/fallback de operacao)
- Comandos de gate no fim do trabalho:
  - `PYTHONPATH=src python3 -m calosum.harness_checks`
  - `PYTHONPATH=src python3 -m unittest discover -s tests -t .`

## Progress

- [x] Leitura do report externo e mapeamento dos 5 pontos.
- [x] Validacao contra documentacao oficial do projeto.
- [x] Triagem de aplicabilidade por ponto.
- [x] Plano em sprints criado em `docs/exec-plans/active/`.
- [ ] Execucao das sprints.
- [ ] Gate final de testes e harness.

## Decision Log

- 2026-03-31: Ponto 1 (runtime persistente) nao e greenfield; decisao de tratar como hardening.
- 2026-03-31: Ponto 2 (MCP/SKILL) aprovado com adocao incremental e opcional para manter estabilidade do core.
- 2026-03-31: Ponto 3 (subagentes) aprovado somente com limites operacionais e budget explicito.
- 2026-03-31: Ponto 4 (prompt-as-data) aprovado como melhoria de modularidade e experimentacao.
- 2026-03-31: Ponto 5 (hooks) aprovado como formalizacao de extensao sobre base existente de event bus.
