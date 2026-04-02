# Neuroplasticity Loop (alias method: apply_neuroplasticity)

## Purpose
Aplicar ajuste incremental nos parametros do bridge/tokenizer com base no `ReflectionOutcome` do group turn.

## Boundaries
- Camada: `domain/metacognition`.
- Metodo canonico: `CognitiveVariantSelector.apply_neuroplasticity(...)`.
- Nome legado preservado no contrato: `apply_neuroplasticity`.

## Inputs / Outputs
- Input: tokenizer/context compressor + outcome da reflexao.
- Efeito: atualizacao dos hiperparametros de controle (ex.: temperatura, ganhos de empatia, limiares).

## Runtime Notes
- O ajuste e deliberadamente conservador (lerp/steps pequenos) para evitar oscilacao abrupta.
- Persistencia entre sessoes e responsabilidade de adapters/bootstrap.
