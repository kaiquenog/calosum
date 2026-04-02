# AgentBaseline Benchmark (Sprint 0)

Data de baseline: 2026-04-02

## Stack
- Agente: `AgentBaseline` (`bootstrap/wiring/agent_baseline.py`)
- Modelo de resposta: LLM API (OpenAI-compatible)
- Embeddings: endpoint de embeddings (OpenAI-compatible ou fallback lexical)
- Memória: JSONL local (`baseline_memory.jsonl`)
- Execução de tools: loop básico via `ConcreteActionRuntime`
- Escopo intencional: sem hemisférios, sem bridge adaptativo, sem group turns

## Métricas obrigatórias

| Métrica | Como medir | Ferramenta | Baseline Sprint 0 |
|---|---|---|---|
| Qualidade de resposta | LLM-as-judge em 50 turnos fixos (escala 1-5) | GPT-4o como juiz | A medir na primeira rodada oficial |
| Taxa de sucesso de tools | `tool_success_rate` por turno | Telemetria do baseline (`payload.tool_success_rate`) | Instrumentado |
| Latência p50/p95 | Tempo total de turno | OpenTelemetry / logs de benchmark | Instrumentado |
| Coerência de sessão | Avaliação humana em 10 sessões multi-turno | Avaliação manual documentada | Pendente rodada humana |

## Protocolo de comparação para próximos sprints
1. Rodar o mesmo conjunto fixo de 50 turnos do baseline.
2. Capturar as quatro métricas acima para o candidato e para o `AgentBaseline`.
3. Reportar deltas absolutos e percentuais.
4. Sprint só pode afirmar melhoria se superar baseline em qualidade sem regressão crítica em `tool_success_rate` e p95.

## Observações
- Este baseline existe para evitar regressões mascaradas por complexidade arquitetural.
- Toda proposta cross-cutting deve incluir comparação explícita contra este baseline.
- O gate automático de CI usa baseline versionado em `docs/benchmarks/ci/baseline.json` para validar regressão de `tool_success_rate` (limite: 5%).
