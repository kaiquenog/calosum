# Title
Execucao do Relatorio de Sinergia OpenClaw + MiroFish

## Purpose
Transformar o relatorio `docs/reports/2026-03-29-synergy-analysis-openclaw-mirofish.md` em trabalho executavel sem aumentar a entropia do repositorio. O objetivo nao foi copiar o relatorio literalmente, mas verificar cada recomendacao pratica contra o estado real do codigo, implementar os ganhos de maior alavancagem e registrar explicitamente o que foi adiado por ja existir, por nao caber no harness atual ou por depender de infraestrutura nao presente.

## Scope
O trabalho foi executado em fases curtas, cada uma fechada com validacao mecanica.

### Fase 0: Auditoria do Relatorio
- Verificada cada recomendacao pratica do relatorio contra o estado real do codigo.
- Classificados os itens como `gap_real`, `parcialmente_implementado` ou `adiado`.
- Registradas as decisoes de escopo para evitar reimplementar mecanismos ja existentes.

### Fase 1: Quick Wins P0 Ainda Reais
- Implementado failover do hemisferio esquerdo em `adapters/llm_failover.py` atras do `LeftHemispherePort`.
- Evoluido o verifier para modo schema-aware com taxonomia de falhas via `shared/schemas.py`.
- Evoluido o `NightTrainer` para um compilador offline `OPRO-lite` reaproveitado pelo hemisferio esquerdo.

### Fase 2: Refinamento P1 de Baixa Entropia
- Substituidas variantes genericas por personas cognitivas explicitas (`analitico`, `empatico`, `pragmatico`).
- Enriquecida a telemetria do hemisferio esquerdo com `system_directives`, preservando as diretivas do `bridge_packet` no dashboard.

### Fase 3: Validacao Operacional
- Rodado `PYTHONPATH=src python3 -m calosum.harness_checks`.
- Rodado `PYTHONPATH=src python3 -m unittest discover -s tests -t .`.
- Reiniciados os containers do `deploy/docker-compose.yml`.
- Validada a API dentro do container `orchestrator` (`/health`, `/ready`, `/v1/chat/completions`, `/v1/telemetry/dashboard/{session_id}`).
- Correlacionados os resultados da API com logs de `orchestrator` e `qdrant`.

### Auditoria Final do Relatorio
- `LLM Failover`: `implementado`.
  Resultado: existe adapter resiliente no bootstrap, mas o ambiente Docker atual nao ativa o failover porque nao define endpoint secundario.
- `OPRO / aprendizado continuo`: `implementado como OPRO-lite`.
  Resultado: o mock foi substituido por compilacao offline heuristica de instrucoes + few-shots.
- `Verifier schema-aware`: `implementado`.
  Resultado: ha validacao tipada e taxonomia (`schema`, `safety`, `runtime`, `incomplete`) propagada ao repair loop.
- `ToolRegistry`: `ja existia`.
  Resultado: o relatorio estava parcialmente defasado; o registry foi mantido como base atual.
- `Knowledge Graph`: `ja existia parcialmente`.
  Resultado: `knowledge_triples` e consolidacao semantica continuam em vigor; `nano-graphrag` ficou fora desta sprint.
- `Personas cognitivas`: `implementado`.
  Resultado: o branching automatico agora usa personas explicitas em vez de variantes genericas.
- `pymdp / VFE`: `adiado`.
  Resultado: permanece fora desta entrega para nao misturar reparametrizacao perceptiva com resiliência e observabilidade.
- `ToolRegistry + code_execution/http_request/browser_read`: `adiado_parcial`.
  Resultado: o registry base foi mantido; `browser_read` continua fora por stack ausente.
- `Channel adapter Telegram`: `adiado`.
  Resultado: segue sem demanda operacional e sem contrato de canal no dominio.

## Validation
- `PYTHONPATH=src python3 -m calosum.harness_checks`
- `PYTHONPATH=src python3 -m unittest discover -s tests -t .`
- `docker compose -f deploy/docker-compose.yml restart`
- `docker compose -f deploy/docker-compose.yml ps`
- `docker compose -f deploy/docker-compose.yml logs --tail=60 orchestrator`
- `docker compose -f deploy/docker-compose.yml logs --tail=60 qdrant`
- Chamadas reais:
  - `/health` -> `{"status":"ok"}`
  - `/ready` -> `{"status":"ready"}`
  - `/v1/chat/completions` -> branching metacognitivo confirmado, com personas `analitico`, `empatico`, `pragmatico`
  - `/v1/telemetry/dashboard/synergy-validation-2` -> `thought.system_directives`, `decision.action_types` e `reflection.selected_variant_id` persistidos corretamente

## Progress
- [x] Fase 0: auditoria do relatorio e definicao do escopo executavel.
- [x] Fase 1: quick wins P0 implementados.
- [x] Fase 2: refinamento P1 de baixa entropia implementado.
- [x] Fase 3: validacao operacional em containers concluida.

## Decision Log
- 2026-03-29: O relatorio foi aceito como insumo estrategico, nao como backlog literal. Itens ja existentes no codigo nao foram reimplementados.
- 2026-03-29: `ToolRegistry` e `knowledge_triples` ja existiam no repositorio; a sprint refinou em vez de duplicar mecanismos.
- 2026-03-29: `pymdp`, `nano-graphrag`, `browser_read` e `Telegram` ficaram explicitamente fora desta execucao para preservar foco e reduzir risco arquitetural.
- 2026-03-29: O verifier schema-aware foi implementado por meio de `shared/schemas.py`, preservando a pureza do dominio e a compatibilidade com o harness.
- 2026-03-29: A validacao operacional precisou ser executada de dentro do container `orchestrator`, porque o sandbox local nao conseguia abrir a porta publicada `127.0.0.1:8000`; a API, ainda assim, respondeu corretamente no ambiente real de execucao.
- 2026-03-29: Durante a validacao em runtime, foi identificado e corrigido um gap de observabilidade: as `system_directives` do `bridge_packet` nao estavam chegando ao `thought` do dashboard quando o hemisferio esquerdo usava `QwenLeftHemisphereAdapter`.
