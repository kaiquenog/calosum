# Calosum

O **Calosum** é um esqueleto para a construção de um Agente de Inteligência Artificial Neuro-simbólico. Ele foi estruturado simulando partes do córtex cerebral humano, dividindo a responsabilidade do agente em diferentes "regiões", como um hemisfério emocional e um lógico, conectados por uma ponte cognitiva.

Atualmente, o repositório é um **alicerce arquitetural com integrações reais parciais**. A base de código, os contratos e a infraestrutura já estão prontos; o hemisfério esquerdo pode falar com OpenAI ou endpoints OpenAI-compatible, e a memória vetorial já usa Qdrant com embeddings configuráveis e fallback explícito quando a stack local ou remota não estiver disponível.

## Arquitetura Base

- **Hemisfério Direito**: Processa as percepções, entende sinais multimodais (áudio, texto, vídeo) e mapeia emoções ou a "saliência" (urgência) do input.
- **Corpo Caloso (`Bridge`)**: Traduz as emoções e o estado latente em "soft prompts" (dicas cognitivas) para direcionar como o outro hemisfério vai reagir.
- **Hemisfério Esquerdo**: Realiza o raciocínio rigoroso, consulta as memórias e elabora um plano de etapas lógicas e seguras.
- **Memória Dual**: O agente possui memórias episódicas (curto prazo) e regras consolidadas (longo prazo).
- **Run-time e Telemetria**: Todo ciclo cognitivo é traçado, metrificado e executado com tentativas de correções (retries automáticos).

---

## Pré-requisitos

Para explorar o projeto localmente, você precisará de:
- `python3 >= 3.11`
- `docker` e `docker compose` (Apenas se quiser rodar serviços opcionais de infraestrutura, como telemetria Jaeger e Qdrant)

---

## Início Rápido: Vendo o Calosum Funcionar

Mesmo sem depender de uma API de Inteligência Artificial pesada, você pode invocar a engine de arquitetura do Calosum no seu terminal para entender como ela divide e soluciona problemas. 

Se a stack opcional de HuggingFace local não estiver disponível, o bootstrap faz fallback automático para o hemisfério direito heurístico em vez de falhar na inicialização.

### 1. Rodar um "Turno" (Um pensamento do agente)

Ao mandar uma frase para o sistema, você verá o modelo capturar as nuances da mensagem e devolver um JSON rico, demonstrando o passo de raciocínio lógico (no hemisfério esquerdo) embasado na percepção de urgência calculada pelo modelo.

Mande um comando avisando que precisa de ajuda com persistência:

```bash
python3 -m calosum.bootstrap.cli run-turn \
  --session-id demo \
  --text "Estou muito ansioso e preciso de um plano urgente!" \
  --infra-profile persistent
```

**Resultado Esperado:** O sistema imprimirá um JSON onde demonstrará a telemetria, sinalizando prioridade de empatia (`empathetic_priority: true`), e o Plano gerado (nos `actions`) que guia o usuário a uma solução calma e gradativa.

### 2. Rodar Cenários Completos via JSON

Você também pode passar um conjunto de falas (turnos) para o agente treinar e rodar reflexões em massa usando os exemplos da pasta de cenários se presentes, ou comandos diretos.

```bash
python3 -m calosum.bootstrap.cli run-scenario caminho/do/cenario.json \
  --memory-dir .calosum-memory \
  --otlp-jsonl .calosum-telemetry/events.jsonl
```

### 2.1. Usando OpenAI no hemisfério esquerdo

Se quiser trocar temporariamente o endpoint local por OpenAI oficial, configure no `.env`:

```bash
CALOSUM_LEFT_ENDPOINT="https://api.openai.com/v1"
CALOSUM_LEFT_API_KEY="sua-chave"
CALOSUM_LEFT_MODEL="gpt-5-mini"
```

Com esse formato, o adapter autodetecta a OpenAI oficial e usa `Responses API` com Structured Outputs. Para workloads mais pesados, troque o modelo para `gpt-5.4`.

### 2.2. Usando embeddings reais na memória vetorial

Se o Qdrant estiver ativo, voce pode configurar um backend de embeddings dedicado:

```bash
CALOSUM_VECTORDB_URL="http://localhost:6333"
CALOSUM_EMBEDDING_ENDPOINT="https://api.openai.com/v1"
CALOSUM_EMBEDDING_API_KEY="sua-chave"
CALOSUM_EMBEDDING_MODEL="text-embedding-3-small"
```

Sem essas variaveis, o builder reaproveita a configuracao OpenAI do hemisferio esquerdo quando ela estiver presente. Se nenhum backend remoto ou local estiver disponivel, o adapter cai para um embedding lexical deterministico, preservando a busca vetorial sem falha dura.

### 3. Experimentos de Ciclo Cognitivo e Reflexão

Na pasta `examples/`, você encontra scripts focados em demonstrar as features específicas de fluxo interno do agente.

```bash
PYTHONPATH=src python3 examples/cognitive_cycle.py
```

---

## Infraestrutura: Rodando com Docker Compose

O agente já possui infraestrutura containerizada planejada para os passos futuros da evolução do sistema.

```bash
docker compose -f deploy/docker-compose.yml up --build -d
```

**O que este comando inicia?**
- `orchestrator` (Levanta o servidor vivo estruturado via FastAPI acessível em `http://localhost:8000/v1/chat/completions`)
- `qdrant` (Banco Vetorial oficial atuando na memória episódica).
- `otel-collector` e `jaeger` (Para tracing e telemetria de performance local).

Você também pode agora rodar sessões de chat infinitas localmente pelo terminal via:
```bash
python3 -m calosum.bootstrap.cli chat
```

O REPL usa a sessao `terminal-session` por padrao. Quando voce sobe a API e a UI localmente sem configuracao adicional, a telemetria desse chat passa a ser persistida em `.calosum-runtime/telemetry/events.jsonl`, permitindo que o painel visualize os turnos do terminal.

---

## Como Começar a Desenvolver (Onde Modificar?)

O projeto adota o estilo **Ports and Adapters**. Você não precisa quebrar as lógicas internas caso queira acoplar a API do ChatGPT ou do LangChain, basta implementar a Interface!

1. Vá em `src/calosum/ports.py` para entender as interfaces que devem ser respeitadas.
2. Para entender as interfaces atuais, veja `src/calosum/shared/ports.py`.
3. Para plugar um LLM de verdade no raciocínio, você pode implementar `LeftHemispherePort` e injetar o adapter pelo construtor em `src/calosum/bootstrap/factory.py`.
4. Todo conhecimento consolidado fica nos Manuais, leia `docs/ARCHITECTURE.md` para entender as barreiras e `docs/PLANS.md` para documentar evoluções maiores.

### Como Realizar Ajustes Finos (Fine-Tuning e ML)

Se o seu objetivo é treinar o agente, ajustar pesos (LoRA/PEFT) ou modificar o comportamento dos embeddings e modelos fundacionais, o **conhecimento mais importante é o isolamento da camada de ML**:

1. **A Regra de Ouro (Isolamento de Tensores):** Todo o código de Machine Learning (`torch`, `transformers`, `peft`) **deve viver exclusivamente na camada `adapters/`**. O núcleo do sistema (`domain/`) não conhece tensores, apenas tipos nativos e contratos (`shared/`).
2. **Treinamento Contínuo (Sleep Mode):** Os ajustes finos contínuos e atualizações de pesos LoRA acontecem em rotinas específicas, como as implementadas em `src/calosum/adapters/night_trainer.py`.
3. **Percepção e Embeddings (Text-JEPA):** Para calibrar a extração de "saliência" (urgência/emoção) do hemisfério direito, altere `src/calosum/adapters/right_hemisphere_hf.py`. Para a memória vetorial do Qdrant, os backends de embedding ficam em `src/calosum/adapters/text_embeddings.py`.
4. **Raciocínio Lógico (LLM):** Mudanças de system prompt, extração estruturada e injeção de "soft prompts" do corpo caloso residem nos adapters do Hemisfério Esquerdo (ex: `src/calosum/adapters/llm_qwen.py`).
5. **Governança (Harness):** Qualquer tentativa de importar bibliotecas de ML diretamente no `domain/` ou `shared/` será bloqueada pelo validador arquitetural `harness_checks.py`. Sempre passe os dados através das interfaces (Ports).

---

## Testes e Validação Arquitetural

Sempre que modificar as pastas do sistema ou arquivos, certifique-se de que a estrutura semântica dos imports e governança mantêm a pureza exigida, rodando o check:

```bash
PYTHONPATH=src python3 -m calosum.harness_checks
```

Para garantir que a base mock e engine rodem perfeitamente, invoque os testes unitários da pasta `tests`:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -t .
```
