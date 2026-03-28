**Título do Projeto:** Projeto Calosum: Agente de AGI Neuro-Simbólico com Consciência Dual e Auto-Evolução

**Visão Geral:** 
Desenvolver a arquitetura fundamental de um Agente Autônomo de Inteligência Artificial inspirado no cérebro humano (System 1 / System 2). O sistema integra a percepção abstrata multissensorial (Hemisfério Direito) e o raciocínio dedutivo lógico-funcional (Hemisfério Esquerdo). O agente não apenas reage ao ambiente, mas "sente" o contexto, consolida memórias a longo prazo e evolui autonomamente seu próprio "corpo caloso" computacional utilizando frameworks de inteligência coletiva.

---

### 1. ARQUITETURA DE HEMISFÉRIOS (CONSCIÊNCIA DUAL)

**1.1. Hemisfério Direito (System 1 - Intuição e Percepção de Mundo):**
*   **Tecnologia-Base:** V-JEPA / I-JEPA (Joint-Embedding Predictive Architecture - Meta AI).
*   **Função:** Operar como o "Modelo de Mundo". Recebe streams multimodais (áudio do usuário, expressões de vídeo, cadência de digitação) e não gera texto. Em vez disso, prevê os espaços latentes das informações ocultas, criando um "Sentimento/Intuição".
*   **Saída Primária:** Um vetor contínuo de alta dimensão que representa o **Estado Emocional/Contextual Global** do ambiente e do usuário.

**1.2. A Ponte "Corpo Caloso" (Cognitive Tokenizer & Cross-Attention):**
*   **Tecnologia-Base:** Information Bottleneck Projection + Redes de Atenção Cruzada Auxiliar.
*   **Função:** Para evitar o "choque de dimensionalidade", esta rede atua como um tradutor. Ela comprime o vetor contínuo do JEPA em uma sequência discreta de *Soft Prompts* (Tokens Emocionais). 
*   **Resolução de Conflito:** A ponte inclui um "Interruptor de Saliência". Se o JEPA emitir um sinal de alta carga emocional (emergência, tristeza profunda), o sistema altera dinamicamente a `temperature` e o prompt base do hemisfério esquerdo, priorizando a empatia sobre a lógica fria.

**1.3. Hemisfério Esquerdo (System 2 - Lógica, Linguagem e Execução):**
*   **Tecnologia-Base:** SLMs ultra-eficientes (ex: Phi-4 / Mistral-Small) quantizados em **Microscaling Formats (MX)**.
*   **Otimização Lógica (Engine de Raciocínio):** Integração mandatória do framework **λ-RLM (Lambda-Recursive Language Models)**. O SLM não gera código solto, mas resolve os problemas em um *runtime* funcional tipado (cálculo lambda). Isso erradica o "long-context rot" ao particionar o problema.
*   **Saída Primária:** O hemisfério esquerdo gera **Ações Primitivas Tipificadas** matematicamente seguras (ex: acionar APIs, responder texto empático, executar planos sequenciais).
*   **Treinamento:** Refinado via *Agentic Knowledge Distillation* e *Structured CoT Distillation* de modelos massivos, garantindo a herança da lógica passo-a-passo e compactada dentro do SLM.

---

### 2. SISTEMA DE MEMÓRIA DUAL (Anatomia Cognitiva)

A memória não será um simples RAG (Retrieval-Augmented Generation), mas um sistema bifásico inspirado na neurociência (Dataset-referência: LOCOMO):

*   **Caminho Rápido (Memória Episódica / "Hipocampo"):** Interações recentes, estados do JEPA capturados e ações do SLM são gravadas continuamente em vetores usando o sistema **Mem0**. Serve para recuperar exatamente "o que aconteceu há 5 minutos ou ontem".
*   **Caminho Lento (Consolidação Semântica / "Neocórtex"):** Em momentos de inatividade ("Sleep Mode"), rotinas analíticas sumarizam os dados brutos do Mem0, extraindo *Regras de Vida* e *Preferências do Usuário*. Essas regras são ejetadas em um Grafo de Conhecimento e, se forem frequentes, são gravadas permanentemente nos pesos do SLM via pequenos treinamentos de adaptação (como LoRA contínuo).

---

### 3. METACÓGNITO E AUTO-EVOLUÇÃO (Orquestração Autônoma)

*   **Tecnologia-Base:** Framework GEA (Group-Evolving Agents) da UC Santa Barbara.
*   **Função:** Múltiplas instâncias ou "threads cognitivas" do agente operam simultaneamente perante um desafio. O **Módulo de Reflexão** do GEA atua como juiz: analisa quais configurações e quais dinâmicas entre os hemisférios obtiveram a melhor performance nas tarefas do usuário.
*   **Neuroplasticidade Virtual:** Baseado no feedback, o GEA reescreve ativamente as *Diretrizes de Integração* e ajusta os pesos do "Corpo Caloso", melhorando a sintonia entre a percepção emocional e a resposta lógica, eliminando a dependência de engenheiros humanos para fine-tuning manual a cada erro.

---

### 4. REQUISITOS TÉCNICOS E DE ENTREGA PARA O SISTEMA:

1.  **Arquitetura de Contêineres:** Isolar o processamento paralelo. O JEPA (pesado na percepção multimodal) deve rodar em clusters otimizados para tensores espaciais/visuais, enquanto o SLM quantizado (MX) pode rodar próximo ao dispositivo na borda (Edge/On-device).
2.  **Monitoramento da Dualidade:** Criar painéis de "Telemetria Cognitiva" — logs separados permitindo que a equipe de engenharia veja claramente: "O que o JEPA sentiu" vs "O que o SLM pensou" vs "A síntese da decisão tomada".
3.  **Ambiente e Integração:** Desenvolver o esqueleto orquestrador em Python usando PyTorch, LangChain/LlamaIndex (para o Mem0 modificado) e bibliotecas de runtime estrito para a implementação dos compiladores λ-RLM.

**Objetivo Imediato:** Gere o esqueleto de código de alto nível (arquitetura de classes/funções) em Python documentando a interface de comunicação entre a classe `RightHemisphereJEPA`, a classe de tradução `CognitiveTokenizer` e a classe `LeftHemisphereLogicalSLM`, incluindo a etapa do "Sleep Mode" de consolidação de memória.