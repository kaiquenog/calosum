# Production Roadmap

Roadmap de estabilizacao e evolucao do Calosum. Atualizado em: 2026-04-03.

## 2026-Q1 (Concluido)

### Arquitetura e Governanca
- [x] Estabelecer Ports and Adapters com verificacao AST mecanica (`harness_checks.py`).
- [x] Harness checks: artefatos obrigatorios, links, planos, tamanho de modulo, fronteiras de importacao.
- [x] Planos versionados em `docs/exec-plans/` como artefatos de primeira classe.

### Hemisferios e Cognicao
- [x] Hemisferio direito: HuggingFace com fallback gracioso (`right_hemisphere_hf.py`).
- [x] Hemisferio direito: V-JEPA 2.1 local-first (`right_hemisphere_vjepa21.py`).
- [x] Hemisferio direito: VL-JEPA multimodal (`right_hemisphere_vljepa.py`).
- [x] Hemisferio direito: backend Rust JEPA-rs (`right_hemisphere_jepars.py`).
- [x] Hemisferio esquerdo: RLM recursivo com fallback para Qwen (`left_hemisphere_rlm.py`).
- [x] Corpus caloso: cross-attention aprendida com `BridgeFusionPort` (`bridge_cross_attention.py`).
- [x] Active Inference refinado com EFE multi-horizonte e novelty density.
- [x] Bidirectional dissonance feedback via workspace compartilhado.

### GEA e Metacognicao
- [x] Experience sharing persistente por SQLite (`gea_experience_store.py`).
- [x] Experience sharing distribuido via Redis (`gea_experience_distributed.py`).
- [x] Reflexao GEA com historico de experiencias (`gea_reflection_experience.py`).
- [x] Variant Preference Model em producao com fallback seguro (`variant_preference.py`, Sprint 5).
- [x] DSPy night trainer para auto-otimizacao de prompts (`night_trainer_dspy.py`).
- [x] LoRA night trainer para fine-tuning local (`night_trainer_lora.py`).

### Observabilidade e Contratos
- [x] Contract wrappers por hemisferio com telemetria de ajuste (`contract_wrappers.py`).
- [x] Backend resolver centralizado para feature flags (`backend_resolvers.py`).
- [x] Idle foraging epistemico para turnos ociosos (`idle_foraging.py`).
- [x] Self-model e introspection capabilities por snapshot (`self_model.py`, `introspection_capabilities.py`).
- [x] Diferential/fuzzy logic com LTN simplificado (`differentiable_logic.py`).
- [x] Telemetria OTLP (`telemetry_otlp.py`).

## 2026-Q2 (Execucao)

- [x] **CI remota com gates:** `mypy --strict`, `ruff`, `harness_checks`, unit + coverage gate, integration gate p95 e benchmark regression gate em `.github/workflows/ci.yml`.
- [x] **Benchmark continuo no CI:** artefatos automaticos em `docs/benchmarks/ci/` e baseline versionado para `tool_success_rate`.
- [x] **Docker local com JEPA:** compose com `CALOSUM_MODE=local`, `CALOSUM_JEPA_MODEL_PATH` e volume read-only do modelo.
- [ ] **Extracao de modulos criticos:** reduzir modulos perto/acima do limite de 400 linhas (`orchestrator.py`, `llm_qwen.py`, `agent_execution.py`, `metacognition.py`).
- [ ] **OTLP para collector externo:** hardening do exportador e teste em stack Docker completa com carga sustentada.
- [x] **Validação de docstrings no harness:** docstrings de módulo obrigatórias nos `__init__.py` de pacotes semânticos (`missing_package_docstring` em `harness_checks.py`).

## 2026-Q3 (Horizonte)

- [ ] Latencia e custo de memoria em perfis Docker otimizados.
- [ ] Regressao automatizada para cenarios multiagente/GEA com dataset fixo.
- [ ] AI-OS: GEA como scheduler de tarefas de fundo com priorizacao epistemica.
- [ ] Dashboard de arquitetura/awareness na UI refletindo estado real de backends.
