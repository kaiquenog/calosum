# Title
Plano de Fechamento Aspiracional: Machine Learning & Integração Contínua

## Purpose
Preencher a lacuna entre a fundação orquestral atual do Calosum (já validada) e a visão AGI original descrita em `INIT_PROJECT.MD`. Focaremos na transição dos "mocks" para modelos de inteligência reais, implementando o pipeline de embeddings para o Hemisfério Direito (restrito a texto, conforme restrição do usuário) e o pipeline de LoRA no Sleep Mode.

## Scope
O trabalho será dividido em Sprints ordenadas da complexidade mais baixa para a mais alta:

### Sprint 1: Integração de Embeddings Reais no JEPA (Baixa Complexidade)
*Substituir o mock de hash por uma inferência de embedding real para capturar a semântica do texto.*
- **1.1 Dependências:** Adicionar `sentence-transformers` ou `HuggingFaceEmbeddings` ao `requirements.txt`.
- **1.2 Adapter do Hemisfério Direito:** Criar o `HuggingFaceRightHemisphereAdapter` que implementa `RightHemispherePort`.
- **1.3 Fluxo Textual:** O adapter deve receber o `user_turn.user_text`, gerar um vetor denso real (ex: `all-MiniLM-L6-v2` de 384 dimensões).
- **1.4 Classificação de Sentimento (Stub Neural):** Usar um modelo zero-shot simples (como `facebook/bart-large-mnli`) ou heurística vetorial para inferir as emoções reais do texto (`emotional_labels`), substituindo a aleatoriedade atual.

### Sprint 2: A Ponte de Atenção Cruzada "Leve" (Média Complexidade)
*Substituir a regra estática de saliência por uma projeção vetorial aprendida (Information Bottleneck simples).*
- **2.1 Camada de Projeção:** Criar uma rede PyTorch minúscula (`nn.Linear`) no `CognitiveTokenizer`.
- **2.2 Tradução:** Essa rede pegará o vetor de 384 dimensões do Hemisfério Direito e o projetará em um vetor de "pesos" que selecionará ativamente quais `Soft Prompts` (ex: diretrizes de segurança vs velocidade) devem ser ativados.
- **2.3 Salvamento de Pesos:** Configurar o orquestrador para carregar os pesos dessa ponte a partir do disco, preparando o terreno para a neuroplasticidade.

### Sprint 3: Framework λ-RLM Estrito (Média-Alta Complexidade)
*Tornar a execução do hemisfério esquerdo funcional de verdade no padrão lambda.*
- **3.1 Parsing de AST:** Atualizar o `StrictLambdaRuntime` para que o `lambda_program` gerado pelo Qwen não seja apenas uma string ignorada. Ele deve ser um Python AST (Abstract Syntax Tree) restrito.
- **3.2 Sandboxing:** Criar um ambiente de avaliação restrito (`ast.literal_eval` evoluído) que permita compor `PrimitiveActions` dentro de loops e condicionais lógicas definidas pelo LLM.

### Sprint 4: Neuroplasticidade via LoRA Contínuo - Sleep Mode (Alta Complexidade)
*Treinamento não supervisionado nos pesos do modelo usando o Qdrant.*
- **4.1 Preparação de Dataset:** Evoluir o `SleepModeConsolidator` para, além de gerar `SemanticRule`s, exportar um arquivo `.jsonl` formatado para fine-tuning (padrão Alpaca/ShareGPT), contendo as interações de maior sucesso do dia (avaliadas pelo GEA).
- **4.2 PEFT/LoRA Integration:** Criar um script isolado `night_trainer.py` que importe a biblioteca `peft` e `transformers`.
- **4.3 Ciclo Noturno:** O script deve carregar os pesos do modelo Qwen, aplicar o adaptador LoRA usando o dataset recém-criado, treinar por poucas epochs (simulando o "sonho") e salvar os novos pesos na pasta do container.
- **4.4 Hot-Reload:** O `QwenLeftHemisphereAdapter` deve ser modificado para verificar e carregar os novos pesos LoRA no início da manhã (próxima inicialização da API).

## Validation
- O Painel de Telemetria (UI) deve exibir as emoções detectadas pela rede neural (e não mais mocks) na coluna *Right Hemisphere*.
- O `harness_checks.py` deve garantir que as novas dependências de ML (`torch`, `transformers`) fiquem isoladas apenas na camada de `adapters`.
- A geração do arquivo LoRA deve ser validada por testes unitários no `SleepModeConsolidator`.

## Progress
- [x] Sprint 1: Integração de Embeddings Reais no JEPA
  - [x] Dependências (`sentence-transformers`, `torch`) instaladas.
  - [x] `HuggingFaceRightHemisphereAdapter` criado.
  - [x] Injeção de dependência atualizada no `CalosumAgentBuilder`.
  - [x] Testes unitários e `harness_checks.py` passando.
- [x] Sprint 2: A Ponte de Atenção Cruzada "Leve"
  - [x] Criada rede `nn.Linear` (Information Bottleneck) no `CognitiveTokenizer`.
  - [x] Adicionada capacidade de carregar `bridge_weights.pt` e realizar inferência (`_neural_translate`).
  - [x] Fallback mantido caso PyTorch não esteja presente ou dimensão do vetor seja diferente (`_heuristic_translate`).
- [x] Sprint 3: Framework λ-RLM Estrito
  - [x] Atualizado `StrictLambdaRuntime` para usar `ast.parse` no código gerado pelo modelo.
  - [x] Sandboxing de segurança impedindo injeção de imports dentro do raciocínio lambda do agente.
- [x] Sprint 4: Neuroplasticidade via LoRA Contínuo
  - [x] Atualizado `SleepModeConsolidator` para exportar JSONL compatível com fine-tuning (ShareGPT/Alpaca style).
  - [x] Instaladas bibliotecas de PEFT e Datasets.
  - [x] Criado o script isolado `night_trainer.py` para injetar os adaptadores de forma programática.

## Decision Log
- Decidiu-se limitar a entrada do JEPA a texto (focado em NLP embeddings) a pedido do usuário, simplificando substancialmente a Sprint 1, evitando o overhead de clusters V-JEPA pesados de vídeo/áudio neste momento.