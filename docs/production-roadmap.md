# Production Roadmap

Roadmap de estabilização e evolução do Calosum. Atualizado em: 2026-03-31.

## 2026-Q1 (Concluído)

### Arquitetura e Governança
- [x] Estabelecer Ports and Adapters com verificação AST mecânica (`harness_checks.py`).
- [x] Harness checks: artefatos obrigatórios, links, planos, tamanho de módulo, fronteiras de importação.
- [x] Planos versionados em `docs/exec-plans/` como artefatos de primeira classe.

### Hemisférios e Cognição
- [x] Hemisfério direito: HuggingFace com fallback gracioso (`right_hemisphere_hf.py`).
- [x] Hemisfério direito: V-JEPA 2.1 local-first (`right_hemisphere_vjepa21.py`).
- [x] Hemisfério direito: VL-JEPA multimodal (`right_hemisphere_vljepa.py`).
- [x] Hemisfério direito: backend Rust JEPA-rs (`right_hemisphere_jepars.py`).
- [x] Hemisfério esquerdo: RLM recursivo com fallback para Qwen (`left_hemisphere_rlm.py`).
- [x] Corpus caloso: cross-attention aprendida com `BridgeFusionPort` (`bridge_cross_attention.py`).
- [x] Active Inference refinado com EFE multi-horizonte e novelty density.
- [x] Bidirectional dissonance feedback via workspace compartilhado.

### GEA e Metacognição
- [x] Experience sharing persistente por SQLite (`gea_experience_store.py`).
- [x] Experience sharing distribuído via Redis (`gea_experience_distributed.py`).
- [x] Reflexão GEA com histórico de experiências (`gea_reflection_experience.py`).
- [x] DSPy night trainer para auto-otimização de prompts (`night_trainer_dspy.py`).
- [x] LoRA night trainer para fine-tuning local (`night_trainer_lora.py`).

### Observabilidade e Contratos
- [x] Contrato wrappers por hemisfério com telemetria de ajuste (`contract_wrappers.py`).
- [x] Backend resolver centralizado para feature flags (`backend_resolvers.py`).
- [x] Idle foraging epistêmico para turnos ociosos (`idle_foraging.py`).
- [x] Self-model e introspection capabilities por snapshot (`self_model.py`, `introspection_capabilities.py`).
- [x] Diferential/fuzzy logic com LTN simplificado (`differentiable_logic.py`).
- [x] Telemetria OTLP (`telemetry_otlp.py`).

## 2026-Q2 (Próximas Prioridades)

- [ ] **CI remota:** Executar harness checks e suite de testes em cada PR (GitHub Actions ou similar).
- [ ] **Extração de módulos críticos:** Reduzir módulos que atingiram o limite de 400 linhas (`orchestrator.py`, `llm_qwen.py`, `agent_execution.py`, `metacognition.py`).
- [ ] **Benchmark contínuo:** Comparativo de latência, surprise médio e taxa de sucesso entre backends JEPA vs HF vs legado.
- [ ] **Checkpoint HF real carregado localmente:** Substituir modo simulado no hemisfério direito por modelo pré-carregado.
- [ ] **OTLP para collector externo:** Hardening do exportador e teste em stack Docker completa.
- [ ] **Validação de docstrings no harness:** Exigir docstrings mínimas nos `__init__.py` de pacotes semânticos.

## 2026-Q3 (Horizonte)

- [ ] Latência e custo de memória em perfis Docker otimizados.
- [ ] Regressão automatizada para cenários multiagente/GEA com dataset fixo.
- [ ] AI-OS: GEA como scheduler de tarefas de fundo com priorização epistêmica.
- [ ] Dashboard de arquitetura/awareness na UI refletindo estado real de backends.

