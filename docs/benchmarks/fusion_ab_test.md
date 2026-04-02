# Fusion A/B Test (Sprint 4)

Data: 2026-04-02
Status: rodada executada (10 turnos) com judge externo; gate estatistico ainda nao atingido.

## Objetivo
Validar se a fusao JEPA-guided melhora coerencia semantica versus baseline sem fusao e versus controle randomico.

## Condicoes
| Condicao | Configuracao |
|---|---|
| Controle | `CALOSUM_FUSION_ENABLED=false` |
| Tratamento A | `CALOSUM_FUSION_ENABLED=true`, `CALOSUM_FUSION_CANDIDATES=3`, `CALOSUM_FUSION_SELECTION_MODE=guided` |
| Tratamento B | `CALOSUM_FUSION_ENABLED=true`, `CALOSUM_FUSION_CANDIDATES=3`, `CALOSUM_FUSION_SELECTION_MODE=random` |

Trigger de custo extra: fusao so dispara quando `jepa_uncertainty < 0.5`.

## Metricas
- Qualidade semantica: LLM-as-judge (escala 1-5) sobre o mesmo conjunto de turnos.
- Taxa de sucesso de tools: `tool_success_rate`.
- Latencia p50/p95 por turno.

## Hipotese
- H1: Tratamento A > Controle.
- H2: Tratamento A > Tratamento B.
- Criterio estatistico: p < 0.05.

## Resultado Atual
- Execucao: `PYTHONPATH=src ./.venv/bin/python3 scripts/fusion_ab_test.py --limit 10 --judge-max-retries 1 --judge-throttle-s 0 --output-json docs/reports/fusion_ab_test_2026-04-02.json`
- Judge model: `gpt-5-mini`
- Controle: score medio `1.0`, p50 `2362.539 ms`, p95 `5135.09 ms`
- Tratamento A: score medio `1.0`, p50 `2401.696 ms`, p95 `4912.705 ms`
- Tratamento B: score medio `1.0`, p50 `2340.698 ms`, p95 `4843.311 ms`
- Estatistica:
  - A vs Controle: `mean_diff=0.0`, `p=1.0`
  - A vs B: `mean_diff=0.0`, `p=1.0`
- Gate Sprint 4: **nao passou** nesta rodada.

Observacao: para conclusao formal do gate com o protocolo completo, falta rodada de 50 turnos e confirmacao de p<0.05.

## Telemetria Esperada
Em `left_result.telemetry`:
- `fusion_enabled`
- `fusion_selection_mode`
- `fusion_method`
- `fusion_triggered`
- `fusion_candidate_count`
- `fusion_temperatures`
- `fusion_selected_index`
- `fusion_uncertainty`
- `fusion_score` e `fusion_scores` (quando `guided`)
