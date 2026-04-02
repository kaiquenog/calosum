# Sprint 2 - JEPA Trained Predictor (MLP), Incerteza e Integracao Local
## Purpose
Substituir a heuristica de predicao JEPA da fase 1 por um preditor treinado real (MLP) para mapear embeddings de contexto em embeddings de resposta boa, com estimativa de incerteza via MC-dropout e fallback explicito para o adapter heuristico.

## Scope
- Introduzir `TrainedJEPAAdapter` no hemisferio direito com:
  - encoder base `all-MiniLM-L6-v2` (frozen, sem fine-tuning)
  - preditor MLP de entrada `[batch, 3, 384] -> [batch, 384]`
  - estimativa de incerteza por MC-dropout (`n_samples=10` por default)
  - telemetria JEPA (`prediction_method`, `jepa_uncertainty`, `surprise_source`)
- Integrar `trained_jepa` no bootstrap resolver:
  - default em modo local (`CALOSUM_MODE=local`) quando backend estiver `auto/legacy/""`
  - fallback para `HeuristicJEPAAdapter` quando checkpoint/dependencias nao estiverem disponiveis
- Criar pipeline de treino/versionamento em `scripts/` e checkpoint versionado em `adapters/jepa_predictor/v1.0/`.
- Registrar metadados de treino (fontes, hiperparametros, metricas, timestamp, semver).

## Validation
- `PYTHONPATH=src ./.venv/bin/python3 -m calosum.harness_checks`
- `PYTHONPATH=src ./.venv/bin/python3 -m unittest tests/adapters/hemisphere/test_right_hemisphere_trained_jepa.py tests/bootstrap/test_factory_backends_2026.py tests/bootstrap/test_settings_dependency_mode.py`
- Smoke de treino (quando dataset preparado):
  - `PYTHONPATH=src ./.venv/bin/python3 scripts/train_jepa_predictor.py --train-jsonl <path> --val-jsonl <path> --output-dir adapters/jepa_predictor/v1.0`

## Progress
- [x] Adapter treinado implementado e coberto por testes.
- [x] Resolver atualizado para default local + fallback heuristico.
- [x] Pipeline de treino e metadados versionados criados.
- [x] Harness e suite focada passando.

## Decision Log
- 2026-04-02: `TrainedJEPAAdapter` sera default apenas em modo local; API mode permanece sem acoplamento em runtime local.
- 2026-04-02: Incerteza sera estimada por MC-dropout para manter compatibilidade com gating de surpresa do pipeline atual.
- 2026-04-02: Fallback heuristico continua como mecanismo de disponibilidade quando o modelo treinado nao estiver pronto.
- 2026-04-02: O checkpoint inicial foi versionado como placeholder (`pending_training`); o treino real atualiza o metadado e gera `predictor.pt`.
