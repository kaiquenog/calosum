# 🚀 Plano de Execução: Transcendência Neuro-Simbólica (Maturidade 2026)

**Data:** 03 de Abril de 2026  
**Baseado no Relatório:** `2026-04-03-calosum-architecture-review-dual-hemisphere.md`  
**Objetivo Primário:** Eliminar as heurísticas fracas e os gargalos de acoplamento do *world model* e planificador lógico. Implementar a matemática rigorosa de *Active Inference* (Free Energy Principle), consolidar a infraestrutura local e fechar o loop de plasticidade em tempo real.

O projeto avançará em 3 Sprints estruturados. O lema desta fase é **"Matemática Real, Não Heurística de Python"**.

---

## 🏃‍♂️ Sprint 1: Fundação EFE e Purificação do JEPA (Crítico)

**Foco:** O alicerce da percepção e a penalidade por falha devem ser matematicamente rígidos.

### Tarefa 1.1: Remover o Fake Fallback no `vjepa21` adapter
- **O que fazer:** Remover as chamadas `np.random.randn(768)` no adapter `InputPerceptionJEPA` e `vjepa21.py` que atuavam como *fallback* na falha de carregamento ou extração da imagem/texto. Corrigir o log-variance (`latent_logvar`) cravado em `-3.0` no V-JEPA.
- **Por quê:** Gerar vetores com ruído para mascarar *exceptions* envenena a infraestrutura do *world model*. A inferência ativa e a incerteza entram em curto-circuito (o modelo acha que entendeu o caos). O `logvar` hardcoded destrói a percepção real de ignorância do sistema.
- **Como:** 
  1. Alterar a camada de `InputPerceptionPort` para ter um estado explícito de ignorância (blindness/degraded mode) retornado via enum/status.
  2. Forçar a variante cognitiva "pragmática" e a dependência do Left Hemisphere em cenários onde a percepção falha.
  3. No caso do V-JEPA local, computar o $\sigma^2$ (variância latente) através de uma inferência de Monte Carlo Dropout, caso a arquitetura não suporte distribuição intrínseca, para estimar incerteza epistêmica de forma limpa.
- **Verificação (DoD):** Teste de integração falha a etapa de percepção sem alucinar vetores randômicos. A telemetria passa a reportar a incerteza real do preditor latente (e não logs falsificados).

### Tarefa 1.2: Racionalização do Roteamento LLM e Fim do HTTP Retry Cego
- **O que fazer:** Substituir o sistema cego de re-tentativas HTTP (`tenacity`) no Left Hemisphere (Action Planner e Runtime).
- **Por quê:** Overengineering com falibilizadores demais (`llm_failover`, `llm_fusion`). O Hemisfério Esquerdo é uma camada simbólica e lógica. Repetir chamadas HTTP iguais num sistema cognitivo é inútil. Erros de execução de *tool* ou *parser* devem imediatamente atualizar o Estado de Crença (Belief State).
- **Como:** 
  1. Implementar o padrão *Recursive Language Models (RLM)* ou Backtracking no Planejador Lógico. Se a execução falha, a árvore de execução atualiza a percepção para "ação inválida" e a política penaliza a heurística.
  2. Deletar `llm_failover` pesado e unificar fallback via uma tabela de roteamento de contexto simples (um LLM pesado + um SLM de emergência).
- **Verificação (DoD):** Quando induzida uma falha de `StrictLambdaRuntime`, a telemetria não mostrará `Retry 1..5`, mas sim um nó de *Backtracking* cognitivo gerando nova instrução $\pi$ atualizada.

### Tarefa 1.3: Refatoração do `math_cognitive.py` (Fundação EFE)
- **O que fazer:** Remover cálculo puramente reativo (Surpresa via similaridade de cosseno) para a função fundamental de *Expected Free Energy (EFE)*.
- **Por quê:** Similaridade de cosseno mede apenas quão distante um conceito está do anterior, não o quão provável ele é e nem a incerteza associada. A EFE vai viabilizar a escolha da política ($G(\pi)$).
- **Como:**
  1. Implementar `calculate_efe(prior_latent, posterior_latent, observation, policy_cost)`.
  2. Decompor em *Risco* (Divergência KL entre posterior e prior) e *Ambiguidade* (Entropia).
- **Verificação (DoD):** Bateria de testes unitários injetando vetores sintéticos com alta variância confirmando que a Ambiguidade força a energia livre total para cima.

---

## 🏃‍♂️ Sprint 2: EFE Distribuído e Pipeline de Memória Causal

**Foco:** Integrar as métricas matemáticas à seleção de personalidade e curar os gargalos de persistência em disco.

### Tarefa 2.1: GEAReflectionController guiado por Inferência Ativa
- **O que fazer:** Substituir os "pesos" hardcoded (0.5 empatia, 0.3 runtime, 0.2 simplicity) pela métrica de EFE.
- **Por quê:** O sistema Multiagente (GEA) tem como núcleo a competição evolutiva de variantes. A seleção não deve ser baseada em pesos lineares viciados e ajustados manualmente no código, mas sim na menor *Expected Free Energy* da política proposta por aquela variante.
- **Como:**
  1. Alterar a assinatura do `aevaluate` no ReflexionController para avaliar a EFE de cada pacote cognitivo candidato `CognitiveCandidate`.
  2. Variantes (analitica/empatica/pragmatica) devem simular a "Surpresa Esperada" caso sua resposta seja enviada, selecionando a que reduz incerteza instrumental e risco epistêmico da conversa (EFE mínima).
- **Verificação (DoD):** Log do GEA não deve mais listar `empathy_score`, mas sim `complexity_penalty` e `ambiguity_cost`. A variante selecionada será sempre a de $min(EFE)$.

### Tarefa 2.2: Transição DuckDB VSS / Otimização do Point-of-Truth
- **O que fazer:** Colapsar as 3 layers de memória concorrentes (`SQL`, `JSONL`, `Qdrant`) para o Hemisfério Lógico e Episódico.
- **Por quê:** O I/O gargala predições do JEPA e infla tempo de resposta total do pipeline FastAPI.
- **Como:**
  1. Substituir a serialização turno a turno em JSONL do `InMemoryEpisodicStore` usando DuckDB com suporte vetorial (VSS). DuckDB opera analítico relacional no disco com extrema performance. 
  2. Tratar a base vetorial (DuckDB ou Qdrant local-only) como *single source of truth* relacional e vetorial.
- **Verificação (DoD):** *Profiling* da aplicação na porta 8000. Redução do tempo *wall-clock* no processo `process_turn()` em pelo menos 40% em perfil `persistent` local.

---

## 🏃‍♂️ Sprint 3: Plasticidade Estrutural e Autonomia de Evolução

**Foco:** Garantir que o Calosum feche o *Loop* entre as camadas e treine automaticamente seus backends à noite.

### Tarefa 3.1: SGD Real no Corpus Callosum (Information Bottleneck)
- **O que fazer:** Abandonar a interpolação linear ingênua (`lerp`) dos pesos base do `CognitiveTokenizer`.
- **Por quê:** O "cérebro" não mistura redes com uma média aritmética. O estrangulamento da informação (`bottleneck`) deve ser guiado via retropropagação autêntica (Stochastic Gradient Descent) da função de perda (minimização de Surpresa).
- **Como:**
  1. Exportar a árvore de execução falha e o desfecho otimizado (Selecionado pelo GEA) como um *dataset* diário.
  2. Implementar a rotina no `NightTrainer` que carrega a matrix de projeção Pytorch (384 -> 64 -> 7), passa a `Loss` (EFE) e dá um `optimizer.step()`.
- **Verificação (DoD):** O arquivo de configuração ou estado binário do `CognitiveTokenizer` demonstra tensores (pesos) atualizados após o comando `sleep` com `NightTrainer`, em direção à melhor extração do *salience neuron*.

### Tarefa 3.2: Fine-tuning Autônomo e Retroativo (Right Hemisphere)
- **O que fazer:** Treino autônomo (backpropagate) na parte inferior da infraestrutura preditiva de estado `V-JEPA-AC`.
- **Por quê:** Até o momento, o *NightTrainer* foca nas "falas" (DSPy/LLM). O modelo de compreensão de mundo (V-JEPA) precisa melhorar e prever o futuro em domínios técnicos que ele nunca viu.
- **Como:**
  1. `SleepModeConsolidator` extrai sessões de "Alto Prediction Error" contínuo (A Incerteza Irredutível).
  2. Acionar uma rotina que gera os pares Preditivos $p(s_{t+k} | s_t)$ e aplica uma micro-sessão de *fine-tuning* (via LoRA/QLoRA) na camada de representação latente do V-JEPA (`predictor`).
- **Verificação (DoD):** Ao rodar o *benchmark script* `right_hemisphere_benchmark.py` e simular 10 dias seguidos de operação em um domínio desconhecido, o `prediction_error` cai consistentemente na telemetria diária, comprovando o aprendizado contínuo estrutural.

---

**Governança de Implantação:** O arquivo `harness_checks.py` deve rodar limpo em cada merge desse plano. Nenhuma ferramenta da Sprint 3 deve ser tentada antes da conclusão das matemáticas de Base da Sprint 1 e 2. O pipeline é um *Directed Acyclic Graph (DAG)* cronológico.