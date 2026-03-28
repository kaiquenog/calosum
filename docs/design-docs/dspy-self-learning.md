# DSPy para Self-Learning (Otimização de Prompts)

## Contexto

Atualmente, o Calosum conta com um mecanismo de "Sleep Mode" projetado para a Fase 4 de evolução do sistema (SOTA - Memória e Inferência Ativa). O objetivo desse modo noturno é realizar a "Destilação Episódica" e o "Treinamento Contínuo", inicialmente idealizado apenas como *fine-tuning* de pesos (LoRA) no arquivo `adapters/night_trainer.py`. No entanto, o ajuste fino de pesos nem sempre é a abordagem mais eficiente, interpretável ou estável para aprimorar o raciocínio lógico (Hemisfério Esquerdo), podendo ser suscetível a "esquecimento catastrófico" (catastrophic forgetting).

## O Que é o DSPy?

O [DSPy](https://dspy.ai/) é um framework declarativo de programação de modelos de linguagem. Ele substitui a engenharia manual de prompts (strings frágeis) por "módulos" programáveis e introduz **Otimizadores** (como `MIPROv2`, `BootstrapFewShot` e `BootstrapFinetune`) que compilam esses módulos.

Os otimizadores do DSPy utilizam um dataset e uma métrica de sucesso para ajustar automaticamente os prompts de uma pipeline. Eles fazem isso explorando instruções em linguagem natural de forma inteligente e sintetizando bons exemplos *few-shot* para maximizar o desempenho do agente em uma tarefa específica.

## Como o DSPy Ajudaria o Calosum? (Self-Learning)

O Calosum gera uma quantidade massiva de telemetria e dados de interações (Memória Episódica), e o `SleepModeConsolidator` já possui a lógica de recuperar esses episódios passados. 

A integração dos otimizadores do DSPy se encaixa de forma nativa e poderosa para resolver o problema de auto-aprendizado (self-learning) do projeto:

1. **Auto-melhoria de Prompts no Sleep Mode:** Durante o modo de descanso, o agente pode utilizar os "turnos bem-sucedidos" da sua memória episódica (episódios em que a ação funcionou de primeira e o usuário ficou satisfeito) como um *Trainset* para o DSPy. O `dspy.MIPROv2` pode rodar em background para descobrir *system prompts* matematicamente mais eficazes do que os escritos originalmente pelo desenvolvedor.
2. **Seleção de Few-Shots:** O otimizador `BootstrapFewShot` pode extrair automaticamente as melhores demonstrações (entradas e saídas perfeitas do passado) e injetá-las no contexto do Hemisfério Esquerdo para o dia seguinte, melhorando a aderência do LLM à estrutura JSON exigida.
3. **Métrica Orgânica:** A robustez do `AgentExecutionEngine` do Calosum (ausência de erros de sintaxe ou auto-correções) e a métrica de "surpresa" (Free Energy) do Hemisfério Direito podem atuar diretamente como a *Metric Function* do DSPy.
4. **Finetuning Dinâmico:** Se a equipe desejar, o otimizador `dspy.BootstrapFinetune` pode orquestrar o fine-tuning de modelos menores localmente, atuando como a evolução madura do plano original de usar LoRA puro.

## Impacto na Arquitetura (Onde modificar no futuro)

O DSPy pode atuar em várias frentes do projeto, substituindo heurísticas e strings fixas por otimização guiada por dados.

### 1. Hemisfério Esquerdo (`adapters/llm_qwen.py`)
O adaptador do LLM seria refatorado para não usar strings fixas, passando a implementar classes `dspy.Signature` e módulos como `dspy.Predict` ou `dspy.ChainOfThought`. A injeção dos "soft prompts" vindos da Ponte seria modelada como campos de entrada no DSPy.

### 2. Modo Noturno / Aprendizado Contínuo (`adapters/night_trainer.py`)
Em vez de ser apenas um loop do PyTorch/PEFT para LoRA, ele instanciaria os otimizadores do DSPy, carregaria o histórico episódico do `memory_qdrant.py` (usando os turnos bem-sucedidos) e salvaria o prompt compilado (ou os pesos, via `BootstrapFinetune`) para o próximo ciclo de execução.

### 3. Ponte Cognitiva e Metacognição (`domain/bridge.py` e `domain/metacognition.py`)
Atualmente, o `CognitiveTokenizer` e o `GEAReflectionController` usam heurísticas e regras fixas para determinar `system_directives` (ex: "lead with empathy before dense logic") e limiares de saliência. 
Com o DSPy, a Metacognição (que já julga qual candidato cognitivo foi melhor avaliado em empatia e segurança) pode atuar como a **função de recompensa (Metric)** para o otimizador. As `system_directives` da Ponte, ao invés de serem *hardcoded*, passariam a ser os "Instructions" que o `MIPROv2` otimizaria continuamente com base nas pontuações geradas pelo juiz GEA.

---
**Conclusão:** O DSPy se apresenta como a ferramenta perfeita para viabilizar a Destilação Episódica e o aprendizado contínuo (Fase 4) do projeto, minimizando o risco de overengineering e maximizando resultados através da otimização estruturada de prompts.
