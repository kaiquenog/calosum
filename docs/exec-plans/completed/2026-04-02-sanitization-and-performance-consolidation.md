# Exec Plan: Sanitização, Performance e Consolidação de Harness

## Purpose
Este plano detalha a transição da arquitetura Calosum de um protótipo de pesquisa acadêmica ("Academic Cosplay") para um sistema de orquestração de IA de nível de produção. O foco é a remoção de over-engineering, redução drástica de latência e fortalecimento da governança arquitetural.

## Scope
1. **Sanitização Cognitiva (Amputação)**:
   - Remover `GEAReflectionController` e a lógica de "Group Turns" (3x LLM calls).
   - Remover dependência `inferactively-pymdp` e simplificar o cálculo de surpresa para uma métrica de distância vetorial pura no `domain`.
   - Desativar `NightTrainerLoRA` para evitar colapso de modelo por feedback loop não supervisionado.
2. **Otimização de Performance**:
   - Consolidar Percepção -> Bridge -> Raciocínio em um fluxo linear único.
   - Implementar "Dynamic Prompt Selector" que altera a system message baseada no estado da Bridge, sem disparar novos processos de inferência.
3. **Isolamento de Infraestrutura**:
   - Mover todas as dependências de ML (`torch`, `transformers`, `peft`) para o perfil opcional `[local]`.
   - Garantir que o `domain/` não contenha tensores ou grafos de computação.
4. **Governança e Harness**:
   - Atualizar `harness_checks.py` para validar a ausência de lógica de treinamento e dependências de ML no core.
   - Expandir a suíte de testes de integração para validar o novo pipeline simplificado.

## Validation
- [ ] **Harness**: `python3 -m calosum.harness_checks` retorna sucesso.
- [ ] **Latência**: Benchmark de turno único em ambiente mockado < 1.5s.
- [ ] **Dependências**: `pip install calosum` (sem extras) não deve instalar `torch` ou `transformers`.
- [ ] **Regressão**: `tests/integration/test_pipeline.py` passando com o novo fluxo.

## Progress
- [ ] Fase 1: Amputação de módulos redundantes (Metacognition/GEA/ActiveInf).
- [ ] Fase 2: Simplificação do Pipeline no Orchestrator.
- [ ] Fase 3: Limpeza de dependências e `pyproject.toml`.
- [ ] Fase 4: Atualização do Harness e validação final.

## Decision Log
- **2026-04-02**: Decisão de remover Group Turns para priorizar latência de usuário.
- **2026-04-02**: Substituição de Active Inference (pymdp) por similaridade de cosseno (EMA) para reduzir overhead matemático desnecessário.
- **2026-04-02**: Congelamento de pesos do modelo (remoção de LoRA autônomo) para garantir estabilidade comportamental.

---

### Detalhamento Técnico das Ações

#### Fase 1: Limpeza de "Academic Bloat"
1. Excluir `src/calosum/domain/metacognition.py` (ou simplificar drasticamente).
2. Remover `inferactively-pymdp` do `pyproject.toml`.
3. Substituir `ActiveInferenceRightHemisphereAdapter` por `SimpleDistanceSurpriseAdapter`.

#### Fase 2: Linearização do Pipeline
1. Alterar `CalosumAgent.aprocess_turn`:
   - De: `if surprise > 0.6: run_group_turn()`
   - Para: `apply_dynamic_personality_injection(bridge_packet)`
2. O "Vencedor" agora é determinado por heurística na Bridge, não por competição de geração.

#### Fase 3: Hardening do Harness
1. Adicionar regra ao `harness_checks.py`: `forbidden_patterns = ["torch.", "nn.Module", "train()"]` em arquivos dentro de `src/calosum/domain/`.
2. Verificar se `shared/` e `domain/` estão limpos de qualquer tipagem que exija `torch.Tensor`.

#### Fase 4: Suite de Testes
1. Criar `tests/test_performance_gate.py` que falha se o pipeline demorar mais que o baseline definido.
2. Garantir que o `MemorySystem` persista triplas semânticas corretamente sem a necessidade do loop de "Sleep Mode" pesado.
