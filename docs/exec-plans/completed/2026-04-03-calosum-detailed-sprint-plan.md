# Plano de Ação Detalhado em Sprints: Evolução Arquitetural Calosum

**Documento de Execução**
**Baseado em**: `2026-04-03-calosum-comprehensive-architecture-review.md`

Este documento detalha o plano de ação cirúrgico para elevar a maturidade do framework Calosum, fechando o abismo entre a teoria arquitetural e a execução prática. O plano é dividido em Sprints focadas, quebrando cada problema diagnosticado em tarefas aplicáveis de engenharia.

---

## Sprint 1: Purificação e Estabilidade (Core Engine Clean-up)
**Objetivo**: Remover lógica simulada destrutiva (mocks), estabilizar o loop cognitivo assíncrono e promover a matemática fundamental (Active Inference) para o Domínio.

### Épicos e Tarefas:

**1.1. Erradicação da Simulação Destrutiva (O Pecado do SHA256)**
* **Contexto**: A geração de pseudo-vetores latentes via SHA256 destrói cálculos de Expected Free Energy (EFE) ao introduzir ruído estocástico no sistema.
* **Ações**:
    * Modificar `vjepa21.py` e `input_perception.py` (e qualquer outro adapter JEPA).
    * Remover a função `_lexical_embedding` (baseada em `hashlib`).
    * Implementar uma representação explícita tipada `NullLatent` (ou disparar interrupção tratada) quando o modelo real não estiver carregado.
    * Garantir que falhas de inferência propaguem incerteza genuína, não "surpresa" artificial.

**1.2. Purificação do Active Inference (Promoção ao Domínio)**
* **Contexto**: O cálculo de Free Energy atua apenas como um wrapper em um Adapter (`ActiveInferenceSurpriseAdapter`), e não como o motor central da política do sistema.
* **Ações**:
    * Mover a matemática do Active Inference dos adaptadores obscuros para o coração do domínio: `src/calosum/domain/cognition/differentiable_logic.py`.
    * Implementar o cálculo rigoroso de `Variational Free Energy com Novelty Weighting Real`, substituindo as heurísticas ingênuas e o uso de "cosine distance" por *KL Divergence* real.
    * *Snippet alvo*: Função `calculate_efe_refined(mu_q, logvar_q, mu_p, logvar_p, epistemic_weight)`.

**1.3. Desbloqueio do Event Bus (Concorrência)**
* **Contexto**: O uso de `await self.event_bus.publish` no loop `process_turn` do Orquestrador causa colapso do sistema se houver listeners bloqueantes, impactando a telemetria e o ciclo cognitivo.
* **Ações**:
    * Otimizar o `event_bus` no Orquestrador.
    * Mudar chamadas bloqueantes para *fire-and-forget* via `asyncio.Queue` (dispatch non-blocking com workers dedicados).
    * Desacoplar completamente a camada de I/O de Observabilidade (OpenTelemetry/Jaeger) da latência cognitiva de percepção e reflexão.

**1.4. Expansão de Governança e Testes**
* **Contexto**: A cobertura de testes falha em auditar os limites matemáticos, e as checagens permitem "números mágicos" travestidos de estatística bayesiana (ex: `return 0.3` no MC-dropout).
* **Ações**:
    * Expandir `harness_checks.py` para auditar e rejeitar "Fallbacks Ruidosos" ou heurísticas fixas de incerteza em contratos de inferência.
    * Criar uma matriz de testes puros em `tests/` para `math_cognitive` (ou `differentiable_logic.py`) focado em provar os limites matemáticos da EFE.

---

## Sprint 2: Recursividade Semântica e Ponte Neural (Left Hemisphere & Callosum)
**Objetivo**: Transformar o processamento do Hemisfério Esquerdo de um partidor de strings para um raciocínio recursivo sobre AST, e estruturar a base de comunicação bidirecional real.

### Épicos e Tarefas:

**2.1. Implementação do Verdadeiro RLM (Recursive Language Models)**
* **Contexto**: O `RlmLeftHemisphereAdapter` usa ingênuo particionamento de texto (`\n\n`) em vez de recursão semântica verdadeira (arXiv 2512.24601).
* **Ações**:
    * Descontinuar o *split lexical* no Hemisfério Esquerdo.
    * Implementar `left_hemisphere_rlm_ast.py`. O modelo deverá operar sobre uma Árvore de Sintaxe Abstrata (AST) de raciocínio.
    * Construir a lógica onde o modelo delega context-windows menores, avalia a completude de nós lógicos (sub-tarefas) e re-executa apenas as sub-árvores que falharam.

**2.2. Reconstrução do Corpus Callosum (Bridge)**
* **Contexto**: A bridge `CognitiveTokenizer` opera de forma anêmica (fluxo Right -> Left), atuando quase como um injetor de prompt sem back-propagation online.
* **Ações**:
    * Refatorar os adapters em `adapters/bridge` (especialmente a Cross-Attention).
    * Estruturar a fundação para uma bridge bidirecional contínua, permitindo *Action Conditioning* onde o *Action Runtime* fornece feedback ao *Preditor Latente*.

---

## Sprint 3: Aceleração Local-First e Multiagente (Performance & GEA)
**Objetivo**: Garantir que o sistema opere em *real-time intuition* (tempo real realístico) em dispositivos de borda, e otimizar os fluxos de experiência multiagente em background.

### Épicos e Tarefas:

**3.1. FFI Nativa em Rust para JEPA (`jepa-rs`)**
* **Contexto**: A inferência pesada baseada puramente em PyTorch/Transformers introduz overhead e bloqueio via GIL, tornando inviável a intuição local em tempo real (destaque para o gargalo do MC-Dropout de 1.5s/turno).
* **Ações**:
    * Criar/integrar os adapters `right_hemisphere_jepars.rs` usando PyO3.
    * Migrar a inferência pesada do V-JEPA para esta biblioteca Rust nativa.
    * Substituir as implementações dependentes do Torch pesado no core de processamento pelo binário otimizado, reduzindo o footprint de memória pela metade.

**3.2. Evolução Multiagente (GEA Daemon)**
* **Contexto**: O `GEAReflectionController` roda baseado em JSONL offline ou com overhead síncrono, quebrando a velocidade iterativa.
* **Ações**:
    * Refatorar o `GEAReflectionController` para atuar como um *Daemon* em background (assíncrono absoluto).
    * Implementar o consumo de um *ring buffer* (DuckDB/Qdrant) de memórias/experiências compartilhadas (arXiv 2602.04837), sem penalizar o `process_turn` do usuário.

**3.3. Otimização de Bloatware de Treinamento (`night_trainer`)**
* **Contexto**: A integração com o DSPy é muito pesada para as características "sleep mode" embarcadas de borda planejadas.
* **Ações**:
    * Aplicar uma dieta drástica nas dependências e na arquitetura do adaptador DSPy (`night_trainer`).
    * Focar a otimização de prompts offline de forma desanexada da base principal em tempo de execução.

---

## Critérios de Sucesso e Validação Final

Ao final deste ciclo de três Sprints, os seguintes critérios devem ser validados pela esteira (harness) e arquitetura de testes:
1. Nenhuma dependência `if-else` ou `haslib` mascarando falhas de ML.
2. Observabilidade emitindo estados em Qdrant/Jaeger estritamente correspondentes à presença/ausência de inferência matemática real.
3. Loops do orquestrador bloqueados por no máximo inferências críticas, com todo sistema de mensageria assíncrono.
4. EFE (Expected Free Energy) orientando o sistema desde o `domain`, coberto por testes matemáticos em `math_cognitive`.
## Purpose

## Scope

## Validation

## Progress

## Decision Log

