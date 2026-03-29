# Title
Evolução Agêntica Prioritária: CRITIC, Tooling Tipado e DSPy

## Purpose
Orientar o próximo ciclo de evolução do Calosum para maximizar ganho prático em produção. O foco deste plano é consolidar auto-correção guiada por ferramentas, endurecer o tool-calling real, transformar o Sleep Mode em auto-otimização via DSPy e usar busca multi-caminho apenas sob orçamento controlado. O plano explicitamente adia investimentos em world models, robotics e embodied AI até que o produto tenha necessidade operacional disso.

## Scope
O trabalho será executado em frentes sequenciadas, com gates entre elas.

### Fase 1: Camada de Crítica e Verificação (CRITIC-like)
**Componentes afetados:** `src/calosum/domain/agent_execution.py`, `src/calosum/domain/metacognition.py`, `src/calosum/domain/orchestrator.py`, `src/calosum/shared/types.py`, `src/calosum/shared/ports.py`
- **1.1 Artefato tipado:** Definir um artefato de critique/verdict sem acoplar SDKs externos ao domínio.
- **1.2 Verificação pós-execução:** Inserir um passo de verificação após a execução das actions e antes da resposta final.
- **1.3 Revisão automática:** Permitir revisão do `LeftHemisphereResult` quando o verificador detectar erro factual, schema inválido, tool mismatch ou unsafe wording.
- **1.4 Persistência:** Persistir feedback crítico em telemetria e memória episódica para reuse posterior.
- **1.5 Cobertura:** Adicionar testes para QA, misuse de tools, structured output inválido e retry loops.

### Fase 2: Hardening de Tool-Calling
**Componentes afetados:** `src/calosum/adapters/action_runtime.py`, `src/calosum/domain/runtime.py`, `src/calosum/shared/types.py`, `src/calosum/shared/ports.py`, `src/calosum/bootstrap/factory.py`
- **2.1 Registry tipado:** Substituir o dispatch ad hoc por um registry tipado de tools com schema de entrada e saída.
- **2.2 Estados explícitos:** Separar claramente `accepted`, `executed`, `rejected` e `needs_approval`.
- **2.3 Policy operacional:** Adicionar policy de permissões, sandbox e limites por tool.
- **2.4 Envelopes normalizados:** Unificar envelopes de erro e resultado para alimentar crítica e retries.
- **2.5 Testes de regressão:** Cobrir seleção de tool, validação, runtime rejection e side effects.

### Fase 3: Sleep Mode com DSPy como padrão
**Componentes afetados:** `src/calosum/domain/memory.py`, `src/calosum/adapters/night_trainer.py`, `src/calosum/adapters/llm_qwen.py`, `docs/design-docs/dspy-self-learning.md`
- **3.1 Dataset mais rico:** Evoluir o `SleepModeConsolidator` para distinguir episódios bons, ruins e corrigidos.
- **3.2 Adapter offline:** Introduzir um adapter de otimização offline baseado em DSPy (`MIPROv2` e `BootstrapFewShot`) sem vazar dependências para `domain`.
- **3.3 Artifacts versionados:** Versionar prompts e few-shots compilados para reload seguro no bootstrap.
- **3.4 LoRA opcional:** Tratar `BootstrapFinetune` e LoRA como complementos posteriores, não como caminho padrão.
- **3.5 Métricas comparativas:** Medir ganho em structured-output validity, `runtime_retry_count`, tool success e delta de latência.

### Fase 4: Busca Multi-Caminho Seletiva (ToT-lite)
**Componentes afetados:** `src/calosum/domain/orchestrator.py`, `src/calosum/domain/metacognition.py`, `src/calosum/domain/right_hemisphere.py`, `src/calosum/adapters/llm_qwen.py`
- **4.1 Gating seletivo:** Ativar branching apenas sob `surprise_score` alto, ambiguidade da tarefa ou tools competitivas.
- **4.2 Orçamento explícito:** Definir limites de profundidade e largura para evitar explosão de custo.
- **4.3 Ranking reutilizado:** Reusar o juiz metacognitivo para ranquear candidatos com critérios observáveis.
- **4.4 Telemetria de custo:** Registrar razões de pruning, vencedor e custo computacional.
- **4.5 Caminho simples preservado:** Validar que tarefas lineares continuam no fluxo padrão.

### Fase 5: Orquestração Multiagente Tardia
**Componentes afetados:** `src/calosum/domain/event_bus.py`, `src/calosum/domain/orchestrator.py`, `src/calosum/bootstrap/`
- **5.1 Entrada condicional:** Só iniciar esta fase após estabilização das Fases 1 a 4.
- **5.2 Papéis mínimos:** Modelar papéis mínimos (`planner`, `executor`, `verifier`) sobre o `InternalEventBus`.
- **5.3 Sem lock-in:** Qualquer integração com AutoGen ou framework similar deve ficar atrás de adapters/bootstrap.
- **5.4 Medição real:** Medir benefício sobre o agente único antes de expandir o número de papéis.

### Fora de Escopo Neste Ciclo
- Integração com `openpi`, `LeRobot`, `Marble` ou outros world models/robotics.
- JEPA action-conditioned, política robótica ou controle físico.
- RL online ou treinamento em tráfego vivo.
- Adoção de dependências pesadas de multiagente antes de evidência de ganho local.

## Validation
- `PYTHONPATH=src python3 -m calosum.harness_checks`
- `PYTHONPATH=src python3 -m unittest discover -s tests -t .`
- Novos testes unitários cobrindo critique loop, registry de tools, artifacts DSPy e gating de ToT.
- Dashboard e telemetria devem expor ao menos: `runtime_retry_count`, `runtime_rejected_count`, `critique_revision_count`, `tool_success_rate`, `branch_count` e `selected_variant_id`.
- Nenhuma dependência nova de DSPy, AutoGen ou SDKs de tools pode entrar em `src/calosum/domain/` ou `src/calosum/shared/`.

## Progress
- [x] Fase 1: Camada de Crítica e Verificação
  - [x] Artefato tipado de critique definido.
  - [x] Verificador inserido no loop principal.
  - [x] Revisão automática conectada ao reparo.
  - [x] Telemetria e memória enriquecidas com feedback crítico.
  - [x] Testes cobrindo regressões principais.
- [x] Fase 2: Hardening de Tool-Calling
  - [x] Registry tipado de tools implementado.
  - [x] Policy de aprovação, permissões e sandbox normalizada.
  - [x] Envelopes de resultado e erro unificados.
  - [x] Testes de runtime e side effects atualizados.
- [ ] Fase 3: Sleep Mode com DSPy como padrão
  - [x] Export de dataset e labels revisado.
  - [x] Adapter offline de otimização criado.
  - [x] Artifacts compilados versionados.
  - [x] Reload seguro de artifacts no runtime do hemisfério esquerdo.
  - [ ] Métricas comparativas documentadas.
- [ ] Fase 4: Busca Multi-Caminho Seletiva
  - [x] Gating por surpresa e ambiguidade definido.
  - [x] Orçamento de branching implementado.
  - [x] Telemetria de pruning e custo adicionada.
  - [ ] Caminho linear preservado e testado.
- [ ] Fase 5: Orquestração Multiagente Tardia
  - [ ] Critérios de entrada para multiagente definidos.
  - [x] Papéis mínimos modelados.
  - [ ] Benefício comparativo medido antes de expansão.

## Decision Log
- DSPy será o caminho preferencial de autoaprendizado antes de LoRA contínuo.
- ToT será seletivo e orientado por orçamento; não ficará no hot path por padrão.
- O próximo grande ganho esperado vem de `CRITIC` e tooling real, não de aumentar a complexidade do hemisfério direito.
- Multiagente será tratado como otimização posterior, não como fundação.
- World models, embodied AI e robotics ficam explicitamente adiados até haver objetivo de produto que justifique sensores, ação física ou simulação 3D persistente.
- O plano permanece ativo porque ainda faltam métricas comparativas do ciclo DSPy, teste explícito do caminho linear sob branching budget e critérios objetivos de entrada para multiagente.
