# Quality Score

Data de referencia: 2026-04-03

## Score Atual

- Orquestracao e contratos: B+
- Memoria e persistencia: B+
- Observabilidade: B
- Harness de governanca: B+
- Integracoes externas reais: B
- Percepcao do Hemisferio Direito: B+
- Conformidade arquitetural: B+

## Critérios

- `A`: pronto para crescimento controlado, com gates e evidencias fortes
- `B`: estrutura boa, mas faltam coberturas adicionais ou CI formal
- `C`: funcional, mas com alto risco de drift
- `D`: placeholder ou sem garantias operacionais

## Justificativas

- **Orquestracao e contratos (B+):** `AgentExecutionEngine` com loop de retry/repair, `VerifierPort`/`CritiqueVerdict`, group turns via `GEAReflectionController`, fallover adapter para left hemisphere. Sprint 0 entregou `CapabilityDescriptor`, `ModelDescriptor`, `ToolDescriptor` e `ComponentHealth`. `contract_wrappers.py` padroniza enforcement por hemisferio.
- **Memoria e persistencia (B+):** `QdrantDualMemoryAdapter` real com embeddings configuráveis, fallback para JSONL e in-memory. `SleepModeConsolidator` integrado. `BridgeStateStore` persistindo estado de neuroplasticidade. Experience sharing via SQLite e Redis.
- **Observabilidade (B):** Telemetria por turno captura `felt`, `thought`, `decision`, `capabilities` e `bridge_config`. `capability_snapshot` passado pelo builder reflete backends e health ao vivo. OTLP adapter presente (`telemetry_otlp.py`) mas exportador externo ainda não hardened em produção.
- **Harness de governanca (B+):** `harness_checks.py` valida artefatos obrigatórios, links em AGENTS.md, refs em docs/index.md, headings de planos, tamanho de módulos (<400 linhas), docstrings de módulo nos `__init__.py` de pacotes semânticos (`missing_package_docstring`), fronteiras de importação via `MODULE_RULES`, isolamento de `shared/` contra imports de `domain` em runtime (`shared_domain_runtime_import`), padrões proibidos em `domain/` (`forbidden_domain_pattern`) e imports diretos de `os`/`subprocess` em adapters (`forbidden_adapter_import`). Ghost rules removidas em 2026-03-31. Lista de códigos: [`docs/references/harness-engineering.md`](references/harness-engineering.md).
- **Integracoes externas reais (B):** Qdrant, HuggingFace, V-JEPA 2.1, VL-JEPA, JEPA-rs, nano-graphrag, DSPy, LoRA, DuckDuckGo, Telegram, Redis (GEA). Subeiu de B- para B com entrega de múltiplos adapters reais.
- **Percepcao do Hemisferio Direito (B+):** Sprints 0-4 concluídas. Backend selecionável por env: `hf`, `vjepa21`, `vljepa`, `jepars`. Active Inference com EFE multi-horizonte e novelty density. Contract wrapper garantindo contrato estável de saída. Melhora de B para B+ pela adição de backends reais e telemetria canônica estável.
- **Conformidade arquitetural (B+):** Fronteiras Ports and Adapters verificadas mecanicamente. 114 testes passando. Harness verde. `backend_resolvers.py` centraliza decisões de routing mantendo `factory.py` limpo. Debito: 5 módulos em zona de risco de tamanho.

## Gaps Prioritarios

- Extração de módulos em zona de risco de 400 linhas (ver `tech-debt-tracker.md`)
- Exportador OTLP hardened para collector externo
- Benchmark do hemisfério direito com checkpoint HF real pré-carregado e dataset ampliado

## Principios para os Sprints

- **Medir antes de construir:** nenhum sprint novo comeca sem benchmark do sprint anterior.
- **Nomes que nao mentem:** cada componente deve ter um nome que descreve o que faz.
- **Interface antes de implementacao:** o Protocol/Port deve ser especificado e revisado antes de qualquer adapter.
- **Fallback sempre:** qualquer componente novo deve ter fallback para o comportamento anterior.
- **Evidencia antes de crenca:** JEPA, fusao e variant selector so vao para producao com A/B test documentado.
- **Uma coisa de cada vez:** sprints focados; se resolver dois problemas, dividir.
