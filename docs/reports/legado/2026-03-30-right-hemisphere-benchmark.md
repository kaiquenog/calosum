# Right Hemisphere Benchmark (2026-03-30)

## Objetivo
Comparar o comportamento do Hemisferio Direito em dois modos:
- `heuristic` (`RightHemisphereJEPA`)
- `embedding_simulated_offline` (`HuggingFaceRightHemisphereAdapter` com patch local sem download)

## Comando Executado
```bash
PYTHONPATH=src .venv/bin/python examples/right_hemisphere_benchmark.py \
  --output-json docs/reports/2026-03-30-right-hemisphere-benchmark.json
```

## Contexto de Execucao
- Data UTC: `2026-03-30T14:31:20Z`
- Dataset curado interno: 8 amostras (`4 high`, `4 neutral`)
- Threshold de classificacao: `salience >= 0.7 => high`
- Ambiente offline: checkpoint Hugging Face nao pode ser baixado; comparativo de embedding foi rodado em modo simulado para validar contrato e comportamento relativo.

## Resultado Resumido

| Adapter | Mode | Accuracy | False Positive (neutral) | Avg Salience | Avg Confidence | Avg Latency (ms) | P95 Latency (ms) | Peak Mem (KB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `RightHemisphereJEPA` | `heuristic` | 1.000 | 0.000 | 0.531 | 0.720 | 0.051 | 0.117 | 6.372 |
| `HuggingFaceRightHemisphereAdapter` | `embedding_simulated_offline` | 1.000 | 0.000 | 0.531 | 0.701 | 0.199 | 0.346 | 6.494 |

## Leitura dos Resultados
- O contrato funcional de percepcao foi preservado nos dois modos para o dataset interno (acerto e falso positivo iguais).
- O modo embedding simulado apresentou latencia maior que o heuristico, como esperado.
- A confianca media do embedding simulado variou com evidencias afetivas; o heuristico manteve valor mais estatico.

## Limites
- Este benchmark **nao** representa desempenho real de embedding com checkpoint oficial, porque o ambiente atual esta offline para download de modelo.
- O dataset ainda e pequeno e curado para smoke validation; ele nao substitui avaliacao ampla em producao.

## Proximo Gate
Para promover o eixo do hemisferio direito para nivel `A`, exigir:
1. benchmark repetido com adapter HF real (checkpoint local pre-carregado),
2. dataset maior e estratificado por linguagem/tonalidade,
3. comparativo em perfis `ephemeral`, `persistent` e `docker` com variancia estatistica.
