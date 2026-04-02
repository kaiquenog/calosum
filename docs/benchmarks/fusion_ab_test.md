# Fusion A/B Test (Sprint 4)

Data: 2026-04-02
Status: protocolo implementado, rodada oficial pendente de execucao com judge externo.

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
- Rodada oficial ainda nao executada neste commit.
- O codigo de fusao, telemetria e toggles de experimento ja estao integrados para execucao do protocolo.

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
