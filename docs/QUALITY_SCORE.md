# Quality Score

Data de referencia: 2026-03-30

## Score Atual

- Orquestracao e contratos: B+
- Memoria e persistencia: B+
- Observabilidade: B
- Harness de governanca: B+
- Integracoes externas reais: B-
- Percepcao do Hemisferio Direito: B

## Critérios

- `A`: pronto para crescimento controlado, com gates e evidencias fortes
- `B`: estrutura boa, mas faltam coberturas adicionais ou CI formal
- `C`: funcional, mas com alto risco de drift
- `D`: placeholder ou sem garantias operacionais

## Justificativas

- **Orquestracao e contratos (B+):** `AgentExecutionEngine` com loop de retry/repair, `VerifierPort`/`CritiqueVerdict`, group turns via `GEAReflectionController`, fallover adapter para left hemisphere. Sprint 0 entregou `CapabilityDescriptor`, `ModelDescriptor`, `ToolDescriptor` e `ComponentHealth` como tipos estáveis em `shared/types.py`.
- **Memoria e persistencia (B+):** `QdrantDualMemoryAdapter` real com embeddings configuráveis, fallback para JSONL e in-memory. `SleepModeConsolidator` integrado. `BridgeStateStore` persistindo estado de neuroplasticidade.
- **Observabilidade (B):** Telemetria por turno captura `felt`, `thought`, `decision`, `capabilities` e `bridge_config`. `capability_snapshot` passado pelo builder reflete backends e health ao vivo. Ainda falta exportador OTLP direto para collector.
- **Harness de governanca (B+):** `harness_checks.py` valida artefatos obrigatórios, links em AGENTS.md, refs em docs/index.md, headings de planos, tamanho de módulos (<400 linhas) e fronteiras de importação via AST em todos os módulos registrados em `MODULE_RULES`. Limite: não valida docstrings de `__init__.py`.
- **Integracoes externas reais (B-):** Qdrant, HuggingFace (right hemisphere), nano-graphrag, DSPy (night trainer), LoRA (night trainer), DuckDuckGo (search_web), Telegram. Fallback gracioso em cada um. Falta CI remota para verificação contínua.
- **Percepcao do Hemisferio Direito (B):** Sprint 0-3 concluídas com contrato estável de telemetria (`right_backend`, `right_model_name`, `right_mode`, `degraded_reason`), calibracao afetiva por threshold de label, confidence dinâmica, componente de novidade (`free_energy_novelty`) e loop bidirecional de feedback operacional via workspace. Sprint 4 adicionou benchmark local reproduzível e relatório em `docs/reports/2026-03-30-right-hemisphere-benchmark.md`. Limite atual: comparativo de embedding ainda em modo simulado offline, sem checkpoint HF real carregado localmente.

## Gaps Prioritarios

- CI remota executando harness checks e testes em cada PR
- Testes de fronteira arquitetural automatizados (regressao de imports)
- Exportador OTLP direto para collector externo
- Validacao de docstrings de `__init__.py` no harness
- Benchmark do hemisferio direito com checkpoint HF real pre-carregado e dataset ampliado
