# Tech Debt Tracker

## Aberto

- Integrar exportador OTLP direto para collector externo (atualmente: JSONL local)
- Adicionar CI remota executando `harness_checks` e testes em cada PR
- Cobrir mais regras arquiteturais via AST (ex: validar docstrings de `__init__.py` por pacote semântico)
- Cobertura de testes de fronteira arquitetural automatizados (regressão de imports)
- Ativar loop de aplicação automática de diretivas `PARAMETER` no awareness loop (Sprint 4)

## Fechado

- ~~Integrar Vector DB real atrás de `MemorySystemPort`~~ — `QdrantDualMemoryAdapter` com fallback JSONL/in-memory (2026-03-28)
- ~~Integrar HuggingFace right hemisphere real~~ — `RightHemisphereHFAdapter` com fallback heurístico (2026-03-28)
- ~~Integrar knowledge graph real~~ — `KnowledgeGraphNanoRAGAdapter` com fallback NetworkX (2026-03-28)
- ~~Implementar failover de left hemisphere~~ — `LLMFailoverAdapter` com primário/secundário (2026-03-28)
- ~~Integrar canal externo (Telegram)~~ — `TelegramChannelAdapter` via `bootstrap/api.py` (2026-03-28)
- ~~Implementar night trainer DSPy~~ — `NightTrainerDSPyAdapter` para otimização de prompts (2026-03-29)
- ~~Implementar night trainer LoRA~~ — `NightTrainerLoRAAdapter` para fine-tuning local (2026-03-29)
- ~~Sprint 0: tipos de capacidade~~ — `CapabilityDescriptor`, `ModelDescriptor`, `ToolDescriptor`, `ComponentHealth` em `shared/types.py` (2026-03-29)
