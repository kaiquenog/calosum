# Title
Autoconsciencia Cognitiva: Introspeccao Arquitetural e Evolucao Autonoma

## Purpose
Transformar Calosum em um agente capaz de introspectar sobre sua propria arquitetura, avaliar seu desempenho historico, diagnosticar gargalos cognitivos e propor diretivas de evolucao concretas — inspirado pelo framework GEA (Group-Evolving Agents, arxiv 2602.04837). O objetivo nao e consciencia filosofica, mas **autoconsciencia operacional**: o agente sabe o que ele e, como funciona, onde falha, e como pode melhorar.

## Contexto: Mapeamento GEA -> Calosum

O paper GEA define autoconsciencia como a capacidade de um agente de:
1. Analisar seus proprios traces de execucao
2. Diagnosticar modos de falha
3. Produzir **diretivas de evolucao** (patches concretos)
4. Aplicar essas diretivas ao proprio framework
5. Compartilhar experiencia entre variantes (group evolution)

Calosum ja possui mecanismos primitivos de cada uma dessas capacidades:

| Capacidade GEA | Calosum Atual | Gap |
|---|---|---|
| Trace analysis | Telemetria 4 camadas (felt/thought/decision/execution) | Nao ha analise sistematica dos traces |
| Failure diagnosis | Verifier com taxonomia de falhas | Nao agrega padroes de falha cross-session |
| Evolution directives | Neuroplasticidade no bridge (metacognition.py) | Limitada a 3 parametros numericos |
| Self-modification | Night trainer (DSPy/OPRO-lite) | So modifica prompts, nao arquitetura |
| Group evolution | GEAReflectionController com 3 personas | Variantes fixas, sem archive evolutivo |

## Scope

### Fase 1: Self-Model — Representacao Interna da Propria Arquitetura

**Objetivo:** O agente precisa ter um modelo de si mesmo — saber quais componentes possui, como estao conectados, e quais parametros sao ajustaveis.

**Tarefas:**

1. Criar `domain/self_model.py` (~200 linhas) com:
   - `CognitiveArchitectureMap`: Dataclass descrevendo a topologia do agente
     - `components`: Lista de `ComponentDescriptor` (name, role, port_protocol, current_adapter, tunable_params)
     - `connections`: Lista de `ConnectionEdge` (source, target, data_type, bottleneck_width)
     - `adaptation_surface`: Dict mapeando parametro -> range valido
   - `build_self_model(agent: CalosumAgent) -> CognitiveArchitectureMap`: Introspecta o agente instanciado via reflection Python (`inspect`, `type()`, `getattr`) e constroi o mapa
   - `diff_models(before, after) -> list[ArchitecturalDelta]`: Compara dois snapshots

2. Adicionar tipos em `shared/types.py`:
   - `ComponentDescriptor`, `ConnectionEdge`, `ArchitecturalDelta`
   - `EvolutionDirective`: (target_component, directive_type, description, priority, estimated_impact)
   - `SelfAwarenessReport`: (architecture_map, performance_profile, diagnosed_bottlenecks, proposed_directives)

3. O self-model nao executa nada — e puramente descritivo. Ele responde perguntas como:
   - "Qual adapter esta implementando meu hemisferio direito?"
   - "Quais parametros do bridge sao ajustaveis e em que range?"
   - "Quantas acoes meu runtime aceita?"
   - "Qual e minha superficie de adaptacao total?"

**Validacao:** `build_self_model()` retorna mapa correto para cada `InfrastructureProfile`.

---

### Fase 2: Introspection Engine — Analise de Performance Historica

**Objetivo:** Analisar traces de telemetria acumulados para identificar padroes de sucesso, falha e estagnacao.

**Tarefas:**

1. Criar `domain/introspection.py` (~300 linhas) com:
   - `IntrospectionEngine`: Consome telemetria historica e produz diagnosticos
   - `analyze_session_history(telemetry_bus, session_id) -> SessionDiagnostic`:
     - Taxa de sucesso do runtime (tool_success_rate medio)
     - Distribuicao de failure_types (schema, safety, runtime, incomplete)
     - Frequencia de repair loops (retry_count medio)
     - Entropia das respostas (diversidade vs repeticao)
     - Eficacia do branching metacognitivo (variante vencedora dominante?)
     - Tendencia de surprise_score ao longo do tempo
   - `diagnose_bottlenecks(diagnostics: list[SessionDiagnostic]) -> list[CognitiveBottleneck]`:
     - `CognitiveBottleneck`: (component, symptom, severity, evidence, suggested_evolution)
     - Exemplos de diagnostico:
       - "Bridge temperature sempre no limite inferior → neuroplasticidade estagnada"
       - "Variante empatica nunca vence → scoring bias ou persona mal calibrada"
       - "Surprise score sempre < 0.3 → percepcao dessensibilizada"
       - "Retry rate > 40% → lambda programs desalinhados com runtime"
       - "Mesmo failure_type em 80% das falhas → verifier gap ou adapter bug"

2. Adicionar `PerformanceProfile` em `shared/types.py`:
   - Metricas agregadas por janela temporal
   - Comparacao com baseline (primeiras N turns vs ultimas N turns)

**Validacao:** Diagnosticos corretos para cenarios sinteticos (mock telemetry com padroes conhecidos).

---

### Fase 3: Evolution Proposer — Diretivas de Auto-Evolucao

**Objetivo:** A partir dos diagnosticos, o agente propoe diretivas de evolucao concretas e priorizadas.

**Tarefas:**

1. Criar `domain/evolution.py` (~250 linhas) com:
   - `EvolutionProposer`: Mapeia bottlenecks para diretivas executaveis
   - `propose_evolution(self_model, bottlenecks, constraint_budget) -> list[EvolutionDirective]`:
     - Cada diretiva e classificada por:
       - `scope`: PARAMETER (ajuste numerico), PROMPT (reescrita de instrucao), TOPOLOGY (troca de adapter), ARCHITECTURE (novo componente)
       - `reversibility`: INSTANT (parametro), SESSION (prompt), RESTART (adapter), MIGRATION (novo modulo)
       - `estimated_impact`: LOW/MEDIUM/HIGH baseado em evidencia historica
     - Regras de proposicao:
       - Bottleneck no bridge → ajuste de adaptation_surface (scope=PARAMETER)
       - Bottleneck na percepcao → sugestao de troca de adapter right hemisphere (scope=TOPOLOGY)
       - Bottleneck no runtime → expansao do executable_actions ou revisao de lambda DSL (scope=PROMPT)
       - Bottleneck na metacognicao → nova persona ou ajuste de scoring weights (scope=PARAMETER)
       - Stagnacao geral → sugestao de novo componente ou integracao (scope=ARCHITECTURE)

2. `EvolutionArchive` em `domain/evolution.py`:
   - Persiste diretivas propostas + resultado observado apos aplicacao
   - Formato: JSONL em `.calosum-runtime/evolution/archive.jsonl`
   - Cada entrada: `{directive, applied_at, observed_before, observed_after, verdict}`
   - Permite aprendizado meta-evolutivo: "diretivas do tipo X tem taxa de sucesso Y"

3. `rank_directives(directives, archive) -> list[EvolutionDirective]`:
   - Score = `estimated_impact * historical_success_rate * (1 / reversibility_cost)`
   - Penaliza diretivas ja tentadas sem sucesso
   - Bonifica diretivas em areas nunca exploradas (novelty bonus, inspirado no Performance-Novelty do GEA)

**Validacao:** Diretivas propostas sao coerentes com os bottlenecks injetados.

---

### Fase 4: Awareness Loop — Integracao no Pipeline Cognitivo

**Objetivo:** O ciclo de autoconsciencia roda periodicamente (nao a cada turn) e alimenta o agente.

**Tarefas:**

1. Adicionar ao `CalosumAgent` em `domain/orchestrator.py`:
   - `self_model: Optional[CognitiveArchitectureMap]` — construido uma vez no boot, atualizado apos neuroplasticidade
   - `awareness_cycle_interval: int = 10` — roda introspeccao a cada N turns
   - `_maybe_run_awareness_cycle(turn_count)`:
     - Se `turn_count % awareness_cycle_interval == 0`:
       - Roda introspeccao sobre as ultimas N turns
       - Diagnostica bottlenecks
       - Propoe diretivas
       - Publica evento `AwarenessCycleEvent` no event bus
       - Aplica diretivas de scope=PARAMETER automaticamente
       - Diretivas de scope >= PROMPT ficam pendentes para aprovacao humana ou sleep_mode

2. Integrar com `asleep_mode()`:
   - Sleep mode agora tambem roda o ciclo de awareness completo
   - Diretivas de scope=PROMPT sao aplicadas durante o sleep (recompilacao via night trainer)
   - Diretivas de scope=TOPOLOGY geram recomendacoes no `ConsolidationReport`

3. Adicionar canal de telemetria `awareness`:
   - Metricas: bottleneck_count, directive_count, auto_applied_count, pending_count
   - Payload: lista de bottlenecks e diretivas propostas
   - Visivel no dashboard

**Validacao:** Ciclo de awareness roda a cada 10 turns, diagnostica corretamente, aplica ajustes parametricos.

---

### Fase 5: Variant Archive — Evolucao Baseada em Populacao

**Objetivo:** Implementar o archive evolutivo do GEA — manter um registro de configuracoes do agente e seus desempenhos.

**Tarefas:**

1. Criar `domain/variant_archive.py` (~200 linhas):
   - `VariantArchive`: Persiste configuracoes de agente + vetor de sucesso
   - `AgentVariantSnapshot`:
     - `variant_id`: UUID
     - `config_hash`: Hash da configuracao completa
     - `bridge_config`: Parametros do tokenizer
     - `persona_weights`: Scores das personas
     - `task_success_vector`: Vetor binario (quais tipos de turn foram bem-sucedidos)
     - `performance_score`: Agregado
     - `novelty_score`: Distancia coseno media aos M vizinhos mais proximos
     - `created_at`, `parent_variant_ids`
   - `select_parent_group(archive, k=2) -> list[AgentVariantSnapshot]`:
     - Score = `performance * sqrt(novelty)` (formula GEA)
     - Retorna top-K

2. Integrar com `GEAReflectionController`:
   - Apos cada group turn, salvar snapshot do vencedor no archive
   - Na proxima reflexao, consultar archive para informar bridge_adjustments
   - Se variante atual esta estagnada (performance plateu), usar parent_group para gerar nova configuracao hibrida

3. Persistencia: `.calosum-runtime/evolution/variant_archive.jsonl`

**Validacao:** Archive cresce monotonicamente, selection prefere variantes com alto score composto.

---

### Fase 6: Conversational Self-Awareness — O Agente Fala Sobre Si

**Objetivo:** O agente consegue responder perguntas sobre si mesmo usando o self-model e introspeccao.

**Tarefas:**

1. Adicionar ao `LeftHemisphereLogicalSLM` em `domain/left_hemisphere.py`:
   - Deteccao de perguntas introspectivas: "como voce funciona", "o que voce sabe fazer", "onde voce esta falhando", "como pode melhorar"
   - Quando detectado, injetar `self_model` e `latest_diagnostics` no contexto de raciocinio
   - Gerar resposta baseada em dados reais, nao em texto generico

2. Nova acao `introspect_self` no runtime:
   - Tipo: `introspect_self`
   - Payload: `{aspect: "architecture" | "performance" | "evolution" | "bottlenecks"}`
   - Retorna: Dados do self-model, diagnosticos recentes, ou diretivas pendentes

3. Isso permite dialogos como:
   ```
   User: "Onde voce esta tendo mais dificuldade?"
   Agent: "Analisando meus ultimos 50 turns: meu taxa de retry esta em 35%,
           concentrada em falhas de schema no lambda program. O bottleneck
           principal e no alinhamento entre as acoes que meu hemisferio
           esquerdo propoe e o que meu runtime aceita. Sugiro expandir
           o vocabulario de acoes ou refinar o prompt de geracao de lambdas."

   User: "O que voce sugere para evoluir?"
   Agent: "Tenho 3 diretivas pendentes ordenadas por impacto:
           1. [PARAMETER] Aumentar salience_threshold de 0.45 para 0.55
              na persona empatica (evidencia: 80% das falhas de safety
              vem dessa variante)
           2. [PROMPT] Reescrever template de lambda program para
              incluir exemplos de acoes validas (estimativa: -20% retry rate)
           3. [TOPOLOGY] Trocar right hemisphere heuristic por HuggingFace
              adapter (surprise score atualmente sem correlacao com
              complexidade real)"
   ```

**Validacao:** Respostas introspectivas contem dados reais da telemetria, nao texto hardcoded.

---

## Roadmap de Evolucao Pos-Autoconsciencia

Uma vez que as 6 fases estejam implementadas, Calosum tera a base para evoluir em direcoes mais ambiciosas. Abaixo estao os eixos de evolucao identificados a partir da analise profunda da arquitetura atual:

### Eixo 1: Percepcao Multi-Modal
- **Estado atual:** Apenas TEXT e parcialmente TYPING/SENSOR
- **Evolucao:** Adapter de percepcao para AUDIO (whisper) e VIDEO (CLIP/SigLIP)
- **Impacto:** Latent vector mais rico → surprise mais preciso → branching mais inteligente
- **Pre-requisito:** Fase 1 (self-model sabe quais modalidades estao ativas)

### Eixo 2: Meta-Aprendizado Arquitetural
- **Estado atual:** Night trainer otimiza prompts
- **Evolucao:** Night trainer tambem otimiza hiperparametros do bridge (via Bayesian optimization ou evolutionary strategies sobre o variant archive)
- **Impacto:** Neuroplasticidade passa de 3 parametros fixos para superficie de adaptacao completa
- **Pre-requisito:** Fase 5 (variant archive com historico de configuracoes)

### Eixo 3: Composicao Dinamica de Personas
- **Estado atual:** 3 personas fixas (analitico, empatico, pragmatico)
- **Evolucao:** Gerar personas sob demanda a partir do self-model + diagnosticos
  - Se bottleneck e safety → gerar persona "guardiao" com temperature=0.05
  - Se bottleneck e criatividade → gerar persona "divergente" com temperature=0.4
- **Impacto:** O agente adapta sua propria diversidade cognitiva
- **Pre-requisito:** Fase 3 (evolution proposer sabe onde estao os gaps)

### Eixo 4: Memoria Autobiografica
- **Estado atual:** Episodica (recente) + semantica (regras) + grafo (triples)
- **Evolucao:** Camada autobiografica: o agente lembra de suas proprias evolucoes, decisoes arquiteturais e o *porque* de cada mudanca
- **Impacto:** Evita regressoes evolutivas (nao tenta diretivas que ja falharam) e constroi narrativa coerente
- **Pre-requisito:** Fase 5 (variant archive) + Fase 2 (introspection engine)

### Eixo 5: Federacao Multi-Agente (GEA Completo)
- **Estado atual:** Um agente com variantes internas
- **Evolucao:** Multiplas instancias de Calosum com archives compartilhados
  - Shared experience pool via Qdrant collection dedicada
  - Parent group selection cross-instance
  - Evolution directives compartilhadas
- **Impacto:** Evolucao em populacao real, nao simulada
- **Pre-requisito:** Todas as 6 fases + infra de comunicacao inter-agente

### Eixo 6: Auto-Verificacao Formal
- **Estado atual:** Verifier heuristico com taxonomia de falhas
- **Evolucao:** Verificacao formal das lambda expressions via type-checking real (mypy-like) e property-based testing (Hypothesis) nos programas gerados
- **Impacto:** Retry rate proximo de zero para falhas de schema
- **Pre-requisito:** Fase 2 (saber onde as falhas ocorrem) + runtime_dsl.py maduro

### Eixo 7: Aprendizado por Curriculo Auto-Gerado
- **Estado atual:** Night trainer compila a partir de episodios passados
- **Evolucao:** O introspection engine identifica *quais tipos de turn* o agente nao sabe resolver e gera exercicios sinteticos para o night trainer
  - Ex: "Retry rate alto em turns com mais de 3 acoes → gerar dataset sintetico de turns complexas para treino"
- **Impacto:** Treinamento direcionado por diagnostico, nao aleatorio
- **Pre-requisito:** Fase 2 + Fase 4 (awareness loop gera diagnosticos periodicos)

### Eixo 8: Ponte Neural Aprendida End-to-End
- **Estado atual:** Information bottleneck 384→64→7 com heuristica como fallback
- **Evolucao:** Treinar a ponte com gradientes reais usando reward signal do verifier
  - Reward = 1.0 se turn aceita sem retry, penalidade proporcional a retries
  - Backprop via REINFORCE ou PPO-lite no bottleneck
- **Impacto:** Bridge aprende mapeamento otimo intuicao→razao
- **Pre-requisito:** Fase 5 (archive com dados de reward por configuracao)

---

## Diagrama da Arquitetura Alvo

```
                    ┌─────────────────────────────────────┐
                    │         AWARENESS LOOP               │
                    │  ┌──────────┐  ┌──────────────────┐ │
                    │  │Self-Model│  │Introspection Eng. │ │
                    │  │  (mapa)  │──│  (diagnosticos)   │ │
                    │  └──────────┘  └────────┬─────────┘ │
                    │                         │            │
                    │  ┌──────────────────────▼──────────┐ │
                    │  │    Evolution Proposer            │ │
                    │  │  (diretivas priorizadas)         │ │
                    │  └──────────┬───────────────────┬──┘ │
                    │             │ auto-apply        │     │
                    │             ▼                   ▼     │
                    │   PARAMETER tweaks    PENDING queue   │
                    └─────────┬───────────────┬────────────┘
                              │               │
          ┌───────────────────▼───────────────▼────────────────┐
          │                 COGNITIVE PIPELINE                   │
          │                                                     │
          │  UserTurn → Right Hemisphere → Bridge → Left Hem.  │
          │      ↓            ↓              ↓         ↓       │
          │  [Event Bus]  [Active Inf.]  [Neural IB] [Lambda]  │
          │      ↓            ↓              ↓         ↓       │
          │  Memory ←── Metacognition ──→ Verifier → Runtime   │
          │      ↓            ↓                        ↓       │
          │  [Episodic]  [GEA Reflect]           [Tool Exec]   │
          │  [Semantic]  [Neuroplast.]                         │
          │  [Graph]     [Var. Archive]                        │
          └────────────────────┬───────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │    SLEEP MODE        │
                    │  Consolidation       │
                    │  Night Training      │
                    │  Full Awareness Cycle│
                    │  Directive Execution │
                    └─────────────────────┘
```

## Validation
- [ ] `PYTHONPATH=src python3 -m calosum.harness_checks` passa em cada fase.
- [ ] `PYTHONPATH=src python3 -m unittest discover -s tests -t .` passa em cada fase.
- [ ] Fase 1: `build_self_model()` retorna mapa correto para perfil ephemeral e persistent.
- [ ] Fase 2: Diagnosticos corretos para cenarios sinteticos com padroes conhecidos.
- [ ] Fase 3: Diretivas coerentes com bottlenecks injetados; archive persiste e recarrega.
- [ ] Fase 4: Awareness cycle roda a cada N turns; parametros ajustados automaticamente.
- [ ] Fase 5: Variant archive cresce; selection respeita Performance*sqrt(Novelty).
- [ ] Fase 6: Agente responde perguntas introspectivas com dados reais.

## Progress
- [ ] Fase 1: Self-Model
- [ ] Fase 2: Introspection Engine
- [ ] Fase 3: Evolution Proposer
- [ ] Fase 4: Awareness Loop
- [ ] Fase 5: Variant Archive
- [ ] Fase 6: Conversational Self-Awareness

## Decision Log
- 2026-03-29: Plano criado inspirado no paper GEA (arxiv 2602.04837). Mapeamento explicito entre conceitos do paper e gaps reais do Calosum.
- 2026-03-29: Self-model e puramente descritivo (read-only sobre a arquitetura) para nao violar as regras de dominio.
- 2026-03-29: Diretivas de scope >= TOPOLOGY requerem aprovacao humana ou sleep_mode; somente PARAMETER e auto-aplicado.
- 2026-03-29: A Fase 5 (Variant Archive) implementa a formula exata do GEA: score = performance * sqrt(novelty).
- 2026-03-29: O roadmap pos-implementacao documenta 8 eixos de evolucao futura, cada um com pre-requisitos explicitos nas fases do plano.
