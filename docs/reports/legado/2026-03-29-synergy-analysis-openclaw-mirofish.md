# Relatório de Sinergia: OpenClaw + MiroFish + Ecossistema → Calosum

**Data:** 2026-03-29
**Escopo:** Análise de como projetos open-source podem acelerar a evolução do Calosum
**Método:** Pesquisa direta nos repositórios + análise da arquitetura Calosum + avaliação honesta de maturidade

---

## 1. Resumo Executivo

O Calosum é uma arquitetura cognitiva neuro-simbólica ~70-75% implementada. Os gaps reais são:

| Gap | Severidade | Estado Atual |
|-----|-----------|--------------|
| Percepção (Right Hemisphere) | ALTA | Heurística determinística, sem modelo treinável |
| Aprendizado contínuo (DSPy/LoRA) | ALTA | Stubs no `night_trainer.py`, nada ativo |
| Ecossistema de ferramentas | MÉDIA | 5 ferramentas base |
| Resiliência LLM | MÉDIA | Sem failover entre provedores |
| Memória estruturada | MÉDIA | Regras textuais, sem grafo de relações |
| Verificação de output | BAIXA | Heurística funcional, mas sem schema |

Este relatório mapeia **soluções concretas** de projetos open-source para cada gap, com avaliação honesta do que vale a pena adotar e do que é hype.

---

## 2. OpenClaw — O que Extrair

### 2.1 Sobre o Projeto

Assistente AI pessoal open-source e self-hosted (100k+ stars). Gateway WebSocket central, 23+ canais de mensageria (WhatsApp, Telegram, Slack, Discord, etc.), plugin system com ClawHub, device control, browser automation. Stack: TypeScript, Node.js 24.

### 2.2 Ideias que Valem a Pena

#### A) Plugin Architecture para Ferramentas

**O problema:** O `ConcreteActionRuntime` tem 5 ferramentas hardcoded. Adicionar cada nova ferramenta exige mexer no adapter.

**O padrão do OpenClaw:** Core mínimo + capacidades como pacotes separados + ClawHub para descoberta dinâmica.

**Proposta prática para o Calosum:**

Não precisa de um framework de plugins. O que resolve é um `ToolRegistry` simples:

```python
# shared/ports.py
class ToolRegistryPort(Protocol):
    def discover_tools(self) -> list[ToolDescriptor]: ...
    def register_tool(self, descriptor: ToolDescriptor, handler: Callable): ...

# adapters/tool_registry.py — scan de diretório com importlib
class DirectoryToolRegistry:
    def __init__(self, tools_dir: Path):
        self.tools_dir = tools_dir
        self._registry: dict[str, ToolHandler] = {}

    def discover_tools(self) -> list[ToolDescriptor]:
        # Cada arquivo em tools_dir/ que tenha TOOL_DESCRIPTOR exportado
        ...
```

Cada ferramenta é um arquivo Python com um `TOOL_DESCRIPTOR` e uma função `execute()`. Sem hot-reload (complexidade desnecessária), sem JSON Schema externo (Pydantic resolve). Migrar as 5 ferramentas existentes e adicionar novas conforme necessidade.

**Ferramentas prioritárias a adicionar:** `code_execution` (sandbox), `http_request` (API genérica), `browser_read` (Playwright para ler páginas).

#### B) LLM Failover

**O problema:** Se o endpoint LLM cai, o Calosum para.

**O padrão do OpenClaw:** Rotação automática entre provedores com fallbacks.

**Proposta prática — NÃO usar litellm:**

litellm é 20k+ linhas e quer controlar toda a chamada LLM. Conflita com o `LeftHemispherePort`. O que o Calosum precisa é um wrapper de ~50 linhas:

```python
# adapters/llm_failover.py
class ResilientLLMAdapter:
    def __init__(self, providers: list[LeftHemispherePort]):
        self.providers = providers
        self._failures: dict[str, int] = {}
        self._cooldowns: dict[str, float] = {}

    async def complete(self, messages, **kwargs):
        for provider in self._healthy_providers():
            try:
                return await provider.complete(messages, **kwargs)
            except (RateLimitError, ConnectionError, TimeoutError):
                self._record_failure(provider)
                continue
        raise AllProvidersDown(...)

    def _healthy_providers(self):
        now = time.time()
        return [p for p in self.providers
                if self._cooldowns.get(p.name, 0) < now]
```

Se futuramente precisar de 10+ provedores, aí sim considerar litellm como backend do adapter. Para 2-3 provedores, o wrapper resolve.

#### C) Channel Adapters

**O problema:** Calosum só é acessível via FastAPI REST + CLI.

**Proposta pragmática:** Começar com **um** canal (Telegram via python-telegram-bot). Um `ChannelPort` em `shared/ports.py` e um adapter. Não construir framework multi-canal antes de ter demanda real para o segundo canal.

```python
# shared/ports.py
class ChannelPort(Protocol):
    async def listen(self, on_message: Callable[[UserTurn], Awaitable[None]]): ...
    async def send(self, session_id: str, text: str): ...
```

#### D) O que NÃO extrair do OpenClaw

- **Gateway WebSocket distribuído** — Overengineering. O Calosum é single-process e deve continuar assim até ter necessidade real de escala.
- **Device control (câmera, tela, GPS)** — Sem caso de uso concreto no Calosum.
- **Voice/TTS** — Prematura. Resolver percepção textual antes de adicionar modalidades.

---

## 3. MiroFish — O que Extrair

### 3.1 Sobre o Projeto

Motor de inteligência de enxame (45k+ stars). Recebe material seed, constrói knowledge graph via GraphRAG, gera milhares de agentes com personalidades, simula interações em plataformas sociais (OASIS/CAMEL-AI), e produz relatórios preditivos. Stack: Python 3.11, Vue.js, GraphRAG, Zep Cloud.

### 3.2 Ideias que Valem a Pena

#### A) Knowledge Graph para Memória — Mas Não o GraphRAG da Microsoft

**O problema:** A memória semântica do Calosum é uma lista de regras textuais. Sem relações entre entidades, sem raciocínio multi-hop.

**Avaliação honesta das opções de Knowledge Graph:**

| Opção | Veredicto | Por quê |
|-------|-----------|---------|
| **microsoft/graphrag** | SKIP | Custo de indexação absurdo ($33K+ para corpora grandes), pipeline pesado, overkill para agente single |
| **nano-graphrag** | RECOMENDADO | ~1100 linhas, NetworkX default (zero infra), hackável, mesma busca local/global do GraphRAG sem o bloat |
| **LightRAG** | ALTERNATIVA | Combina KG com embedding retrieval, ~10x menos tokens que GraphRAG, reranker support |
| **Graphiti (Zep)** | FUTURO | Grafos temporais com janelas de validade — elegante mas requer Neo4j/FalkorDB |
| **NetworkX + SQLite puro** | MÍNIMO VIÁVEL | Se quiser só entity-relationship tracking sem retrieval sofisticado |

**Proposta concreta — nano-graphrag atrás de Protocol:**

```python
# shared/ports.py
class KnowledgeGraphPort(Protocol):
    def add_triple(self, subject: str, predicate: str, obj: str, metadata: dict): ...
    def query_related(self, entity: str, depth: int = 2) -> list[KnowledgeTriple]: ...
    def merge_from_episode(self, episode: EpisodicMemory): ...

# shared/types.py
@dataclass
class KnowledgeTriple:
    subject: str
    predicate: str
    object: str
    confidence: float
    source_episodes: list[str]
```

**Fallback chain:** nano-graphrag → NetworkX + SQLite → dicionário em memória. Mesma filosofia de degradação graceful que o Calosum já usa (Qdrant → JSONL → in-memory).

**Integração no pipeline:**
- `SleepModeConsolidator` gera triples além de regras textuais
- `MemoryContext` passa `relevant_subgraph` para o Left Hemisphere
- Left Hemisphere usa relações para raciocínio contextualizado

#### B) Personas Cognitivas (Evolução Natural do GEA)

**O problema:** O `GEAReflectionController` cria 3 variantes genéricas (conservative/exploratory/creative). São variações de parâmetros, não perspectivas distintas.

**O padrão do MiroFish:** Agentes com personalidades únicas, cada um com perfil, memória, e regras comportamentais.

**Proposta pragmática — sem criar framework de agentes:**

Não precisa do OASIS nem de "milhares de agentes". O que melhora a qualidade é ter 3-5 **personas bem definidas** com `CognitiveBridgePacket` próprio:

```python
COGNITIVE_PERSONAS = {
    "analitico": BridgeControlSignal(
        target_temperature=0.2,
        empathy_priority=False,
        system_directives=["priorize consistência lógica", "identifique contradições"]
    ),
    "empatico": BridgeControlSignal(
        target_temperature=0.6,
        empathy_priority=True,
        system_directives=["considere impacto emocional", "valide sentimentos"]
    ),
    "pragmatico": BridgeControlSignal(
        target_temperature=0.3,
        empathy_priority=False,
        system_directives=["minimize ações", "resposta mais curta possível"]
    ),
}
```

O scoring do GEA já funciona — só precisa receber personas ao invés de variantes genéricas. Sem namespace de memória por persona (overengineering para o momento), sem debate estruturado (complexidade prematura).

**Evolução futura:** Se as personas provarem valor, aí sim adicionar memória per-persona e debate.

#### C) O que NÃO extrair do MiroFish

- **Simulação social com milhares de agentes** — O Calosum não é um motor de previsão. Não tem caso de uso.
- **Plataformas duais (Twitter/Reddit simulados)** — Conceito interessante para pesquisa, irrelevante para agente cognitivo.
- **Zep Cloud como backend de memória** — Dependência de serviço externo. O Qdrant + JSONL do Calosum resolve.
- **GraphRAG da Microsoft** — Já avaliado acima. Overkill.

---

## 4. Percepção: O Gap Mais Crítico (V-JEPA e Alternativas)

O `RightHemisphereJEPA` é 100% heurístico. Pesquisei o estado real do ecossistema JEPA:

### 4.1 Estado do V-JEPA

| Projeto | Estado | Serve para Calosum? |
|---------|--------|---------------------|
| **V-JEPA 2.1** (Meta, facebookresearch/vjepa2) | 1.2B params, HuggingFace Transformers, março 2026 | NAO — vision/video only, precisa de GPU pesada |
| **I-JEPA** (facebookresearch/ijepa) | Estável mas antigo | NAO — image only |
| **eb_jepa** (facebookresearch/eb_jepa) | Limpo, modular, Apache license | FUTURO — mais leve que V-JEPA, mas ainda vision-focused |
| **ijepa-text** (daohanlu/ijepa-text) | Experimental | INTERESSANTE — JEPA para texto, pesquisa-grade |
| **lang-jepa** (jerber/lang-jepa) | Experimental | INTERESSANTE — predição em "concept space" |
| **M3-JEPA** (HongyangLL/M3-JEPA) | ICML 2025, multi-modal MoE | MELHOR CANDIDATO — alinha múltiplas modalidades em latent space via JEPA |
| **TI-JEPA** (ducngg/tijepa) | Texto+Imagem, energy-based | BOM — tem função de energia usável para surprise |
| **`pip install jepa`** (v0.1.0) | Single author, imatura | NAO — arriscado para produção |

### 4.2 Recomendação Prática

**Curto prazo:** Manter o `HuggingFaceRightHemisphereAdapter` com SentenceTransformers. Ele funciona, é leve, e lida com Português. Não trocar por V-JEPA agora — o ganho não justifica a complexidade.

**Melhoria imediata viável:** Substituir o cálculo de surprise heurístico por **Variational Free Energy** real usando **pymdp** (`pip install inferactively-pymdp`). NumPy-only, leve, implementa:
- Variational Free Energy (VFE)
- Expected Free Energy (EFE) para seleção de ações
- Surprise como negative log model evidence
- Loop completo de Active Inference

```python
# adapters/active_inference.py
from pymdp import utils
from pymdp.maths import spm_log_single as log_stable

class ActiveInferenceAdapter:
    """Computa surprise real via VFE ao invés de cosine distance."""

    def compute_surprise(self, observation_vector, generative_model):
        # Discretizar embedding em N estados protótipo (via clustering)
        obs_discrete = self._quantize(observation_vector)
        # VFE = KL[Q(s)||P(s)] - E_Q[log P(o|s)]
        vfe = self._compute_vfe(obs_discrete, generative_model)
        return vfe
```

**Limitação honesta:** pymdp trabalha com estados discretos. Os embeddings contínuos precisam de quantização (clustering em N protótipos). Isso funciona mas adiciona um passo de calibração.

**Médio prazo:** Quando M3-JEPA ou eb_jepa ganharem suporte a texto, avaliar como backend do `RightHemispherePort`. A interface Protocol já suporta a troca sem impacto no domain.

---

## 5. Aprendizado Contínuo: DSPy na Prática

O `night_trainer.py` já tem a estrutura certa. A pesquisa revelou detalhes práticos importantes:

### 5.1 GEPA > MIPROv2 para o Calosum

**GEPA** (novo optimizer padrão do DSPy) é melhor que MIPROv2 para o perfil do Calosum:
- Funciona com **3-10 exemplos** (MIPROv2 precisa de mais)
- Incorpora **feedback textual** — mapeia perfeitamente aos scores do GEA e critique do Verifier
- 93% no MATH benchmark vs 67% baseline
- O backlog do Sleep Mode gera exatamente o tipo de dados que GEPA consome

### 5.2 Integração com OpenAI-Compatible Endpoints

DSPy usa LiteLLM por baixo. Configuração para os endpoints do Calosum:

```python
import dspy

# Endpoint OpenAI-compatible (Qwen, Ollama, vLLM)
lm = dspy.LM(
    'openai/modelo-escolhido',
    api_base=os.environ['CALOSUM_LEFT_ENDPOINT'],
    api_key=os.environ.get('CALOSUM_LEFT_API_KEY', 'placeholder')
)
dspy.configure(lm=lm)
```

Regra: prefixo `openai/` para endpoints genéricos, `ollama_chat/` para Ollama.

### 5.3 Caminho de Implementação em 3 Fases

**Fase 1 — Quick win sem DSPy (1-2 dias):**

Implementar um loop OPRO mínimo no `NightTrainer`: pegar os top 5 episódios bem-sucedidos, montar meta-prompt com prompt+score, pedir ao LLM para propor prompt melhor, avaliar com métrica do GEA, manter o melhor. ~50-100 linhas, zero dependência nova.

**Fase 2 — DSPy com GEPA (1 sprint):**

```python
# adapters/night_trainer.py — substituir os stubs
class DSPyNightTrainer:
    def optimize(self, episodes: list[EpisodicMemory]):
        trainset = [
            dspy.Example(
                user_message=e.user_text,
                context=e.semantic_context,
                response=e.agent_response
            ).with_inputs("user_message", "context")
            for e in episodes if e.quality_label == "good"
        ]

        program = dspy.ChainOfThought("user_message, context -> reasoning, actions")

        def metric(example, pred, trace=None):
            return self._gea_score(pred)  # Reusa scoring do GEA

        compiled = dspy.GEPA(metric=metric, max_iterations=20).compile(
            program, trainset=trainset
        )
        compiled.save(str(self.artifact_dir / "compiled_prompt.json"), save_program=False)
```

**Fase 3 — MIPROv2 para structured output (futuro):**

Só se necessário para otimizar a geração de JSON/ações do Left Hemisphere.

### 5.4 Gotchas Importantes

- **Custo:** Otimização complexa = $20-50 por run. Usar `auto="light"` em dev.
- **Portabilidade:** Prompts otimizados para Qwen **não transferem** para Llama. Re-otimizar ao trocar modelo.
- **Versão:** DSPy < 3.0 não garante compatibilidade de artefatos salvos. Pinar a versão.
- **Split de dados:** 20% treino / 80% validação (contraintuitivo mas previne overfitting). GEPA é exceção — usa split padrão ML.

---

## 6. Verificação de Output: O que Realmente Precisa

### 6.1 Avaliação das Opções

| Opção | Veredicto | Por quê |
|-------|-----------|---------|
| **Instructor** (jxnl/instructor) | BOM, mas desnecessário agora | Faz uma coisa bem (Pydantic + retry). Mas o Calosum já tem repair loop no orchestrator |
| **guardrails-ai** | SKIP | Bloatware. Problemas Pydantic v1/v2, Hub pesado, faz pouco por muita dependência |
| **PydanticAI** | FUTURO | Elegante mas quer controlar o pipeline inteiro |
| **Pydantic puro** | RECOMENDADO | Sem framework. O verifier já faz critique; torná-lo schema-aware com Pydantic é a evolução natural |

### 6.2 Proposta Concreta

O `HeuristicVerifier` já funciona. A evolução é torná-lo schema-based:

```python
# domain/verifier.py — evolução natural
from pydantic import BaseModel, ValidationError

class ActionSchema(BaseModel):
    action_type: str
    typed_signature: str
    parameters: dict

class SchemaAwareVerifier:
    def verify(self, result: LeftHemisphereResult) -> CritiqueVerdict:
        issues = []

        # Verificação existente (unsafe wording, etc.)
        issues.extend(self._heuristic_checks(result))

        # Nova: validação de schema
        for action in result.actions:
            try:
                ActionSchema.model_validate(action.__dict__)
            except ValidationError as e:
                issues.append(f"Schema inválido: {e}")

        return CritiqueVerdict(
            is_valid=len(issues) == 0,
            identified_issues=issues,
            ...
        )
```

Pydantic já é dependência do projeto (FastAPI). Zero overhead.

### 6.3 Padrão LLM-as-Critic que Funciona

A pesquisa converge em: **decomposição de constraints em checks atômicos** (o verifier já faz isso) + **taxonomia de erros** com repair prompts específicos por tipo. O `CritiqueVerdict` poderia categorizar falhas:

```python
class FailureType(Enum):
    SCHEMA_VIOLATION = "schema"      # → repair: "corrija o formato"
    UNSAFE_CONTENT = "safety"        # → repair: "remova linguagem insegura"
    HALLUCINATION = "hallucination"  # → repair: "baseie-se apenas em fatos do contexto"
    INCOMPLETE = "incomplete"        # → repair: "complete a resposta"
```

Cada tipo gera prompt de repair diferente ao invés do genérico "tente novamente".

---

## 7. Matriz de Priorização (Revisada)

Removidos itens de overengineering. Focado no que resolve gaps reais com esforço proporcional.

| # | Melhoria | Fonte | Esforço | Impacto | Prioridade |
|---|----------|-------|---------|---------|-----------|
| 1 | **LLM Failover (~50 linhas)** | OpenClaw | 1-2 dias | ALTO | P0 |
| 2 | **OPRO quick win no NightTrainer** | DSPy research | 1-2 dias | ALTO | P0 |
| 3 | **Verifier schema-aware (Pydantic)** | Instructor research | 1 dia | MÉDIO | P0 |
| 4 | **Surprise via pymdp (VFE real)** | Active Inference research | 1 sprint | ALTO | P1 |
| 5 | **Knowledge Graph (nano-graphrag)** | MiroFish | 2 sprints | ALTO | P1 |
| 6 | **DSPy GEPA no NightTrainer** | DSPy | 1 sprint | ALTO | P1 |
| 7 | **ToolRegistry + 3 novas ferramentas** | OpenClaw | 1-2 sprints | MÉDIO | P1 |
| 8 | **Personas cognitivas no GEA** | MiroFish | 1 sprint | MÉDIO | P2 |
| 9 | **Channel Adapter (Telegram)** | OpenClaw | 1 sprint | MÉDIO | P2 |
| 10 | **Failure taxonomy no Verifier** | CRITIC research | 2-3 dias | MÉDIO | P2 |

**Removidos por overengineering:**
- ~~CognitiveGateway distribuído~~ — sem necessidade real
- ~~OASIS Simulation Engine~~ — Calosum não é motor de previsão social
- ~~Namespaced Memory per-persona~~ — prematuro, validar personas primeiro
- ~~Debate estruturado multi-agente~~ — prematuro, validar personas primeiro
- ~~Browser automation~~ — adicionar quando houver caso de uso

---

## 8. Projetos Complementares (Avaliação Honesta)

### Vale Adotar

| Projeto | O quê | Como | Quando |
|---------|-------|------|--------|
| **pymdp** (`inferactively-pymdp`) | Active Inference, VFE, surprise real | Adapter em `adapters/active_inference.py` | P1 — resolve gap de percepção sem trocar modelo |
| **nano-graphrag** | Knowledge Graph leve (~1100 linhas) | Adapter em `adapters/knowledge_graph.py` | P1 — evolui memória semântica |
| **DSPy** (GEPA optimizer) | Otimização automática de prompts | Substituir stubs em `night_trainer.py` | P1 — ativa aprendizado contínuo |

### Vale Estudar, Não Adotar Agora

| Projeto | O quê | Por quê esperar |
|---------|-------|-----------------|
| **LightRAG** | KG + embedding retrieval | Alternativa ao nano-graphrag se precisar de retrieval mais sofisticado |
| **Graphiti** (Zep) | Grafos temporais com validade | Elegante para memória que envelhece, mas requer Neo4j/FalkorDB |
| **eb_jepa** (Meta) | JEPA modular e leve | Quando ganhar suporte a texto, avaliar para Right Hemisphere |
| **M3-JEPA** (ICML 2025) | Multi-modal JEPA com MoE | Melhor candidato futuro para percepção multi-modal real |
| **Instructor** | Structured output + retry | Útil se precisar extrair dados de texto externo, não para pipeline interno |

### Não Vale a Pena

| Projeto | Por quê |
|---------|---------|
| **microsoft/graphrag** | Custo proibitivo, pipeline pesado, overkill para single-agent |
| **guardrails-ai** | Bloatware com problemas de compatibilidade Pydantic |
| **litellm** (como framework) | 20k+ linhas querendo controlar o call path inteiro. Conflita com Ports & Adapters |
| **LangGraph** | Framework pesado para algo que o orchestrator já faz com retry loops |
| **V-JEPA 2.1** | 1.2B params, GPU pesada, vision-only — não resolve percepção textual |
| **`pip install jepa`** (v0.1.0) | Imatura, single author, sem comunidade |

---

## 9. Roadmap Revisado

### Semana 1-2: Quick Wins (P0)
```
[ ] LLM Failover adapter (~50 linhas em adapters/llm_failover.py)
[ ] OPRO loop mínimo no NightTrainer (~80 linhas, zero dependência nova)
[ ] Verifier schema-aware com Pydantic (evolução do HeuristicVerifier)
```

### Sprint 1-2: Fundações (P1)
```
[ ] pymdp para surprise/VFE real no Right Hemisphere adapter
[ ] DSPy GEPA substituindo stubs no night_trainer.py
[ ] nano-graphrag atrás de KnowledgeGraphPort
[ ] ToolRegistry simples + code_execution + http_request + browser_read
```

### Sprint 3-4: Refinamento (P2)
```
[ ] Personas cognitivas no GEAReflectionController
[ ] Failure taxonomy no CritiqueVerdict com repair prompts tipados
[ ] Channel adapter Telegram
```

### Futuro: Quando Houver Necessidade
```
[ ] Trocar SentenceTransformers por M3-JEPA/eb_jepa (quando suportarem texto)
[ ] Graphiti para memória temporal (quando o KG básico não der conta)
[ ] MIPROv2 para structured output (quando GEPA não resolver)
```

---

## 10. Conclusão

A arquitetura Ports & Adapters do Calosum foi desenhada para absorver essas evoluções — cada melhoria entra como adapter atrás de Protocol, sem tocar no domain core.

**OpenClaw** inspira **infraestrutura prática**: failover LLM, tool registry, channel adapters. Não como framework, mas como padrões de ~50-100 linhas cada.

**MiroFish** inspira **inteligência estruturada**: knowledge graphs para memória relacional e personas cognitivas para diversidade de perspectiva. De novo, sem o peso do OASIS ou simulação social.

**O ecossistema científico** (pymdp, DSPy GEPA, nano-graphrag) fornece as peças que faltam para tornar o pipeline cognitivo real: surprise baseado em VFE, aprendizado contínuo via otimização de prompts, e memória grafada.

O princípio guia: **cada melhoria deve resolver um gap real com esforço proporcional**. Os 3 quick wins da semana 1-2 (~150 linhas no total) já desbloqueiam resiliência, aprendizado básico, e verificação tipada.

---

*Relatório gerado em 2026-03-29. Pesquisa direta nos repositórios GitHub + avaliação da arquitetura Calosum.*

### Fontes Primárias

**Projetos analisados:**
- [OpenClaw](https://github.com/openclaw/openclaw) — Assistente AI self-hosted, 100k+ stars
- [MiroFish](https://github.com/666ghj/MiroFish) — Motor de inteligência de enxame, 45k+ stars

**Projetos recomendados:**
- [pymdp](https://github.com/infer-actively/pymdp) — Active Inference em Python, NumPy-only
- [nano-graphrag](https://github.com/gusye1234/nano-graphrag) — GraphRAG leve (~1100 linhas)
- [DSPy](https://github.com/stanfordnlp/dspy) — Otimização de prompts (GEPA/MIPROv2)
- [LightRAG](https://github.com/hkuds/lightrag) — KG + embedding retrieval

**Projetos avaliados e descartados:**
- [microsoft/graphrag](https://github.com/microsoft/graphrag) — Overkill para single-agent
- [guardrails-ai](https://github.com/guardrails-ai/guardrails) — Bloatware
- [litellm](https://docs.litellm.ai/) — Pesado demais como framework, útil como referência de padrões

**Pesquisa JEPA:**
- [V-JEPA 2.1](https://github.com/facebookresearch/vjepa2) — Vision-only, 1.2B params
- [eb_jepa](https://github.com/facebookresearch/eb_jepa) — JEPA modular (Meta)
- [M3-JEPA](https://github.com/HongyangLL/M3-JEPA) — Multi-modal JEPA (ICML 2025)
- [TI-JEPA](https://github.com/ducngg/tijepa) — Text-Image JEPA com energia
- [ijepa-text](https://github.com/daohanlu/ijepa-text) — JEPA para texto
- [Graphiti](https://github.com/getzep/graphiti) — Grafos temporais (Zep)
