# Pragmatic Refactoring: From Academic Chimera to Production-Ready Agent

## Purpose
O objetivo deste plano é refatorar o Calosum de um "experimento de laboratório super-engenheirado" para um orquestrador de IA pragmático, estável e escalável em produção. O foco principal é remover a complexidade acidental (parsings artesanais de DSL, fake embeddings baseados em hash, e orquestração stateful) que ameaça a estabilidade e a escalabilidade do sistema. Ao final desta execução, o sistema deverá usar padrões de mercado (Structured Outputs/Function Calling), ser *stateless* a nível de aplicação, e manter uma arquitetura "Honesta" em relação às suas capacidades atuais de Neuro-Simbologia e Active Inference.

## Scope
O escopo engloba 5 fases críticas:
1. **Erradicação do StrictLambdaRuntime e DSL LISP:** Substituição por Pydantic/Function Calling nativo.
2. **Transformação Stateless do Orquestrador:** Remoção de estado em memória (dicionários do Python) no CalosumAgent em favor de persistência via DuckDB/Qdrant.
3. **Limpeza da Pseudociência no JEPA e EFE:** Substituição do heuristic_jepa (baseado em hash) por embeddings literais; remoção de matemáticas mágicas de Expected Free Energy.
4. **Isolamento do NightTrainer:** Remoção do DSPy do fluxo autônomo (Background) e transição para uma ferramenta de CLI operada sob demanda (Human-in-the-loop).
5. **Atualização de Documentação e Quality Score:** Alinhamento dos manifestos técnicos com a nova arquitetura simplificada.

## Validation
A validação de sucesso deste plano dependerá de:
- **Testes Unitários:** Falhas de parsing DSL devem zerar (o sistema usará validação Pydantic estrita).
- **Testes de Integração/Carga:** Múltiplos requests simultâneos com o mesmo session_id via FastAPI não devem corromper o Workspace (validação do Orquestrador Stateless).
- **Higiene:** O script harness_checks.py deve continuar passando, garantindo que as fronteiras arquiteturais não foram violadas durante a refatoração.
- **Redução de Código:** Espera-se uma redução líquida de complexidade e linhas de código (deleção do runtime_dsl.py e simplificação do heuristic_jepa.py).
- **Quality Score:** A documentação deve atingir nota máxima nos critérios de Consistência e Simplicidade definidos no QUALITY_SCORE.md.

## Progress
- [ ] **Fase 1: Erradicação da DSL LISP**
  - [ ] Deletar src/calosum/domain/execution/runtime_dsl.py.
  - [ ] Refatorar StrictLambdaRuntime (domain/execution/runtime.py) para aceitar literais Pydantic/JSON baseados no padrão de Function Calling (OpenAI/Qwen).
  - [ ] Atualizar LeftHemisphereLogicalSLM (domain/cognition/left_hemisphere.py) para gerar JSON/Structured Outputs em vez da string LISP.
  - [ ] *Double-check:* O schema de ferramentas (ex: respond_text, search_web) foi exportado via model_json_schema() do Pydantic?
  - [ ] *Test:* Criar um teste injetando um output JSON malformado do LLM para garantir que o fallback/retry do execution_engine lidará graciosamente com o erro estrutural, sem quebrar o AST parser que não existe mais.

- [ ] **Fase 2: Orquestrador Stateless (Escalabilidade Horizontal)**
  - [ ] Remover self.last_workspace_by_session e self.latest_awareness_by_session da classe CalosumAgent em domain/agent/orchestrator.py.
  - [ ] Alterar process_turn para carregar o CognitiveWorkspace do repositório de persistência (DuckDB ou MemorySystemPort) no início do request e salvá-lo ao final do request.
  - [ ] *Double-check:* O cache local de sessão em memória foi completamente extirpado?
  - [ ] *Test:* Escrever um script cliente assíncrono (usando httpx) que dispara 5 requisições concorrentes para a mesma sessão FastAPI. Verificar no banco de dados se os turnos foram processados em sequência atômica (usar locking ou versionamento otimista, se necessário).

- [ ] **Fase 3: JEPA Honesto e Active Inference Realista**
  - [ ] Refatorar adapters/hemisphere/right_hemisphere_heuristic_jepa.py. Remover o código que usa hashlib.sha256 para gerar "vetores latentes".
  - [ ] Alterar o Heuristic JEPA para consumir o provedor de Text Embeddings padrão (text_embeddings.py) ou operar em modo *passthrough* (onde a saliência é determinada unicamente por palavras-chave, sem inventar floats randômicos).
  - [ ] Remover do Orchestrator (orchestrator.py) os cálculos estáticos falsos de Expected Free Energy ((ambiguity_score * 0.6) + (semantic_density * 0.4)). Substituir por um threshold simples e documentado de incerteza baseada na densidade do prompt.
  - [ ] *Double-check:* As métricas reportadas no Jaeger/OTLP mostram "Heuristic Vector" ou "Literal Embedding"?
  - [ ] *Test:* Injetar a mesma frase duas vezes na percepção; a distância cosseno deve ser estritamente 0 (e não uma variação pseudo-aleatória devido a timestamps no hash).

- [ ] **Fase 4: Demover o Night Trainer para Human-in-the-loop**
  - [ ] Remover qualquer gatilho assíncrono que rode o adapters/night_trainer/night_trainer_dspy.py de forma autônoma após X turnos ou no final do dia.
  - [ ] Criar o comando calosum-harness optimize-prompts em bootstrap/entry/cli.py.
  - [ ] *Double-check:* O arquivo compiled_prompt.json só deve ser alterado manualmente ou quando o desenvolvedor explicitamente executar a CLI.
  - [ ] *Test:* Executar o agente por 20 turnos simulados e garantir que não há processos filhos zumbis consumindo CPU/GPU com execuções do DSPy MIPROv2. Rodar a CLI manualmente e verificar o sucesso do pipeline de otimização isolado.

- [ ] **Fase 5: Documentação e Quality Score Alignment**
  - [ ] Atualizar docs/ARCHITECTURE.md: Remover referências à Lambda DSL e ao Strict Runtime LISP. Descrever o novo fluxo de Structured Outputs.
  - [ ] Atualizar docs/RELIABILITY.md: Documentar a nova natureza Stateless do orquestrador e como ele escala horizontalmente.
  - [ ] Atualizar docs/INFRASTRUCTURE.md: Refletir a simplificação dos adapters e a remoção de dependências pesadas de treino autônomo (DSPy) do runtime de API.
  - [ ] *Double-check:* Rodar calosum-harness e garantir que todos os REQUIRED_DOCS e links em index.md estão consistentes com a nova estrutura.
  - [ ] *Test:* Verificar se o arquivo docs/QUALITY_SCORE.md reflete a métrica de "Simplicidade" (redução de LOCs e de camadas de parsing).

## Decision Log
- **[2026-04-02] Decisão de remover DSL Própria:** A criação de uma DSL LISP embutida provou-se frágil contra as falhas estocásticas dos LLMs modernos. A indústria padronizou-se em JSON Schema / Function Calling, que possui validação out-of-the-box (Pydantic), reduzindo o código a ser mantido no nosso repositório em aprox. 15% e aumentando a estabilidade exponencialmente.
- **[2026-04-02] Remoção de Estado no Orquestrador:** Decidido tornar o CalosumAgent stateless para permitir deploys em Kubernetes/ReplicaSets. Manter o dicionário em Python preveniria failover básico e balanceamento de carga real.
- **[2026-04-02] Transparência do Active Inference:** Para não ferir os princípios de integridade da engenharia, métricas teóricas que não contavam com cálculo bayesiano real (fake EFE e fake Latent Vectors) foram substituídas por aproximações de engenharia simples (embeddings clássicos e NLP keywords). Quando os pesos do V-JEPA verdadeiro forem carregados na infra, a métrica complexa será reativada pelos Adapters adequados.
