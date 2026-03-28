# Calosum

O **Calosum** é um esqueleto para a construção de um Agente de Inteligência Artificial Neuro-simbólico. Ele foi estruturado simulando partes do córtex cerebral humano, dividindo a responsabilidade do agente em diferentes "regiões", como um hemisfério emocional e um lógico, conectados por uma ponte cognitiva.

Atualmente, o repositório é um **alicerce arquitetural**. Isso significa que a base de código (o design, os contratos e a infraestrutura) está pronta, possuindo simulações locais (mocks) que demonstram como o fluxo completo de pensamento do agente acontece, mas a integração com provedores reais (como um LLM da OpenAI para o raciocínio ou um banco vetorial Qdrant para memória) deve ser implementada como o próximo passo.

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

---

## Como Começar a Desenvolver (Onde Modificar?)

O projeto adota o estilo **Ports and Adapters**. Você não precisa quebrar as lógicas internas caso queira acoplar a API do ChatGPT ou do LangChain, basta implementar a Interface!

1. Vá em `src/calosum/ports.py` para entender as interfaces que devem ser respeitadas.
2. Para plugar um LLM de verdade no raciocínio, você pode herdar a classe de `ports.LeftHemispherePort` e criar, por exemplo, um `OpenAILogicalSLM(LeftHemispherePort)`. Em seguida, passe esse módulo pelo Construtor (Factory) do agente no `src/calosum/factory.py`.
3. Todo conhecimento consolidado fica nos Manuais, leia as pastas `docs/ARCHITECTURE.md` para entender as barreiras e `docs/PLANS.md` para documentar suas evoluções maiores.

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
