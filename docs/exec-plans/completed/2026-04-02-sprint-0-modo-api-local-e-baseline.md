# Sprint 0.2-0.5 - Modo API/Local, Renomeacoes, Limpeza e Baseline
## Purpose
Executar o gate de Sprint 0 separando modo de runtime (`api` vs `local`), removendo ambiguidade de nomenclatura cognitiva, limpando artefatos de desenvolvimento e estabelecendo `AgentBaseline` para comparação obrigatória dos próximos sprints.

## Scope
- Introduzir `CALOSUM_MODE` com validações explícitas no bootstrap/factory.
- Definir comportamento do night trainer por modo: `api` (DSPy only), `local` (DSPy/LoRA/QLoRA).
- Renomear conceitos centrais para nomenclatura aderente à implementação com aliases de compatibilidade.
- Remover artefatos `pip_install_log.txt`, `test_full_log.txt`, `test_output.txt` e endurecer prevenção via gitignore + pre-commit hook versionado.
- Implementar `AgentBaseline` (LLM API + embeddings + memória JSONL + tool loop básico).
- Documentar baseline em `docs/benchmarks/baseline.md`.

## Validation
- `PYTHONPATH=src ./.venv/bin/python3 -m calosum.harness_checks`
- `PYTHONPATH=src ./.venv/bin/python3 -m unittest tests/bootstrap/test_settings_dependency_mode.py tests/bootstrap/test_factory.py tests/bootstrap/test_agent_baseline.py`
- Verificação manual de ausência de artefatos `.log` e `test_*.txt` fora de `docs/`.

## Progress
- [x] Modo `CALOSUM_MODE=api/local` introduzido em `settings.py`.
- [x] Validações explícitas adicionadas no bootstrap/factory para combinações incoerentes.
- [x] Night trainer atualizado para suportar `lora`/`qlora` em modo local e bloquear em modo api.
- [x] Renomeações semânticas implementadas com aliases de compatibilidade.
- [x] `AgentBaseline` implementado e testado.
- [x] Limpeza de artefatos e hook de pre-commit adicionados.
- [x] Documento de baseline criado.

## Decision Log
- 2026-04-02: Mantidos aliases retrocompatíveis (`CognitiveTokenizer`, `GEAReflectionController`, `apply_neuroplasticity`) para evitar quebra imediata de API/testes.
- 2026-04-02: `CALOSUM_MODE` passa a ser a semântica de execução; `CALOSUM_DEPENDENCY_MODE` permanece por compatibilidade de instalação e é validado contra o modo.
- 2026-04-02: Hook de pre-commit versionado em `.githooks/pre-commit`; ativação via `git config core.hooksPath .githooks`.
