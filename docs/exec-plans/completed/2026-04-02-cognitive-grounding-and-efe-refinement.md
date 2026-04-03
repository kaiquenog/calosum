# 🧠 PLAN: Calosum Cognitive Grounding & EFE Refinement

## Purpose
Este plano visa eliminar as heurísticas superficiais (keyword matching e if/else cognitivos) e substituí-las por implementações matematicamente fundamentadas de **Active Inference**, **Expected Free Energy (EFE)** e **Latent Integration**. O objetivo é levar o framework do nível de "simulação de software" para "arquitetura neuro-simbólica real".

## Scope

### 1. Right Hemisphere: Perceptual De-Symmetrization
- Remover funções de extração emocional baseadas em strings (`_extract_emotional_labels`).
- Implementar **Uncertainty-Aware Embeddings** usando a variância das ativações latentes (ou MC-dropout no adapter treinado).
- Integrar o cálculo de **Context Novelty** baseado na divergência KL entre o turno atual e a distribuição histórica da sessão.

### 2. Active Inference & EFE Logic
- Refatorar `GEAReflectionController` para usar a fórmula de **Expected Free Energy (EFE)**.
- Implementar a decomposição de EFE em:
    - **Ambiguity (Epistemic Value):** Incerteza do modelo sobre o estado.
    - **Risk (Pragmatic Value):** Divergência entre o estado predito e o estado objetivo (priors).
- Definir thresholds de surpresa baseados em desvio padrão móvel, eliminando o valor estático `0.6`.

### 3. Neural Bridge (Corpus Callosum)
- Migrar do `SoftPromptProjector` heurístico para um `LatentProjectionInterpretor`.
- Implementar a projeção de **Continuous Control Signals** que modulam dinamicamente não apenas a temperatura, mas o *top-p* e as *logit_bias* do hemisfério esquerdo com base na salience.

### 4. Left Hemisphere & Runtime Stabilization
- Substituir a gramática `TypedLambdaProgram` customizada por **Structured Outputs (JSON Schemas)** nativos para reduzir o loop de reparo.
- Implementar o `RLM (Recursive Language Model)` local como adapter primário de raciocínio, movendo LLMs de API para fallback de "emergência cognitiva".

## Validation

### Métricas de Sucesso
- **EFE Convergence:** O score de EFE deve diminuir em sessões de longa duração com o mesmo contexto (indicação de aprendizado/estabilização).
- **Tool Success Rate:** Aumento de >15% na taxa de sucesso de primeira tentativa devido ao uso de Schemas nativos.
- **Cognitive Diversity:** Variantes GEA devem ser selecionadas com base na redução real de incerteza (Ambiguity) e não por bias de persona.

### Testes Necessários
- `tests/adapters/perception/test_efe_math.py`: Validar a implementação da divergência KL e entropia.
- `tests/integration/test_latent_bridge_stability.py`: Garantir que vetores de alta variância resultam em sinais de controle conservadores.

## Implementation Steps

### Sprint 1: O Ciclo de Energia Livre (Critical)
1. **Refatorar `shared/models/jepa.py`:** Adicionar campos `latent_mu` e `latent_logvar` ao `InputPerceptionState`.
2. **Atualizar `adapters/perception/active_inference.py`:** Implementar `calculate_efe(mu, logvar, prior_mu, prior_logvar)`.
3. **Modificar `domain/metacognition/metacognition.py`:** Substituir o `linear_pass` por `EFE_minimization_loop`.

### Sprint 2: Limpeza Perceptiva (Sanitization)
1. **Expurgar `adapters/hemisphere/input_perception_trained_jepa.py`:** Remover todo o dicionário `salience_keywords` e a lógica de strings.
2. **Implementar `MC-Dropout` no `TrainedJEPAAdapter`:** Para gerar distribuições de probabilidade (incerteza) reais sobre o vetor latente.
3. **Calibrar `salience`:** Basear o valor na norma do vetor de gradiente (ou magnitude da surpresa) em vez de exclamações no texto.

### Sprint 3: Bridge & RLM (Integration)
1. **Refatorar `domain/cognition/bridge.py`:** Implementar a modulação de parâmetros via `BridgeControlSignal` usando mapeamentos não-lineares.
2. **Promover `action_planner_rlm.py`:** Torná-lo o adapter padrão para perfis `local` e `persistent`.

## Progress
- [x] Pesquisa de constantes de decaimento para EFE Bayesian priors. (100%)
- [x] Refatoração do `InputPerceptionState`. (100%)
- [x] Implementação da função `calculate_efe`. (100%)
- [x] Remoção de heurísticas de string no Right Hemisphere. (100%)

## Summary
- Active Inference foi consolidado com decomposição de EFE (risk + ambiguity), surpresa calibrada por histórico e `context_novelty` por KL sobre episódios recentes.
- `TrainedJEPAAdapter` deixou de usar matching textual para saliência/emotional labels e passou a usar posterior latente (mu/logvar) com MC-dropout.
- Bridge migrou de `SoftPromptProjector` heurístico para interpretação analítica de latente com sinais contínuos de `temperature`, `top_p` e `logit_bias` via anotações de controle.
- Resolver do hemisfério esquerdo passou a promover RLM por default em perfis `local`/`persistent`/`docker`.
- Cobertura de validação adicionada para matemática EFE e estabilidade da bridge com entradas de alta variância.

## Decision Log
- **2026-04-02:** Decidido abandonar a DSL customizada em favor de Schemas Pydantic nativos para aumentar a resiliência do SLM (Small LLM) em execuções locais.
- **2026-04-02:** Decidido que a "Surpresa" será uma métrica relativa (Z-score sobre o histórico da sessão) e não um valor absoluto entre 0 e 1.

---
*Documento gerado pelo Especialista em Revisão de Arquitetura Calosum.*
