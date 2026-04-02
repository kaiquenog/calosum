# trained-jepa-v1.0

Diretorio de artefatos do preditor JEPA treinado (Sprint 2).

Arquivos esperados apos treino:
- `predictor.pt`
- `training_metadata.json`

Comando de treino:
`PYTHONPATH=src ./.venv/bin/python3 scripts/train_jepa_predictor.py --train-jsonl <train.jsonl> --val-jsonl <val.jsonl> --output-dir adapters/jepa_predictor/v1.0 --version 1.0`
