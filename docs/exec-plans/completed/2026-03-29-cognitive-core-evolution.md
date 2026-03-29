# Title
Evolucao do Core Cognitivo: Pendencias de Sinergia (Localhost)

## Purpose
Retomar e implementar os itens arquiteturais propostos no relatorio de sinergia (`docs/reports/2026-03-29-synergy-analysis-openclaw-mirofish.md`) que foram adiados na primeira iteracao. O foco e estritamente no núcleo cognitivo rodando localmente (localhost), preservando a arquitetura de Ports & Adapters e as regras do harness, sem introduzir overengineering.

## Scope
A execucao sera dividida em fases independentes, permitindo validacao isolada de cada capacidade.

### Fase 1: Expansao do ToolRegistry (Novas Ferramentas)
- **Objetivo:** Aumentar a capacidade de atuacao do agente sem ferir o dominio.
- **Tarefas:**
  - Criar ferramenta `code_execution` (sandbox local em Python via `subprocess` com timeout seguro) em `adapters/tools/`. Concluido.
  - Criar ferramenta `http_request` (utilizando `httpx` com timeouts e validacao de schema) em `adapters/tools/`. Concluido.
  - Adicionar as ferramentas ao `ToolRegistry` existente e aos prompts do hemisferio esquerdo. Concluido.
  - *Nota:* `browser_read` com Playwright sera considerado "nice to have" para evitar inflar dependencias prematuramente.

### Fase 2: Percepcao Avancada (Surprise via pymdp)
- **Objetivo:** Evoluir o calculo heuristico de surprise para Variational Free Energy (VFE) real.
- **Tarefas:**
  - Adicionar `inferactively-pymdp` ao `pyproject.toml` e `requirements.txt`. Concluido.
  - Criar um novo adapter `adapters/active_inference.py` que recalcula `surprise_score` por free energy discreta, usando `pymdp` quando disponivel e `NumPy` como fallback compatível. Concluido.
  - Manter o contrato atual do `RightHemispherePort` e integrar o wrapper no bootstrap. Concluido.

### Fase 3: Memoria Semantica Estruturada (nano-graphrag)
- **Objetivo:** Integrar raciocinio multi-hop e relacoes de entidades.
- **Tarefas:**
  - Adicionar `nano-graphrag` as dependencias. Concluido.
  - Criar `adapters/knowledge_graph_nanorag.py` implementando um store de subgrafo com persistencia local e backend compatível com `nano-graphrag`, com fallback `NetworkX`. Concluido.
  - Integrar a geracao de grafos ao `SleepModeConsolidator` sem duplicar extracao de triples, reaproveitando os `KnowledgeTriple` ja gerados pelo dominio. Concluido.
  - Injetar o subgrafo relevante no `MemoryContext` consumido pelo Hemisferio Esquerdo, inclusive em cenarios cross-process via reload do arquivo de grafo. Concluido.

### Fase 4: Aprendizado Continuo (DSPy + GEPA)
- **Objetivo:** Substituir a heuristica OPRO-lite por otimizacao de prompts real via DSPy.
- **Tarefas:**
  - Adicionar `dspy` as dependencias. Concluido.
  - Criar `adapters/night_trainer_dspy.py` implementando um caminho DSPy/GEPA opcional e compatível com o artefato `compiled_prompt.json`. Concluido.
  - Garantir que a configuracao do DSPy respeite o provedor LLM configurado via `.env` / `CALOSUM_LEFT_ENDPOINT`, `CALOSUM_LEFT_PROVIDER` e `CALOSUM_LEFT_API_KEY`. Concluido.
  - Preservar `OPRO-lite` como fallback deterministico para nao perder o ciclo noturno em ambientes sem DSPy. Concluido.

## Validation
- [x] `PYTHONPATH=src python3 -m calosum.harness_checks` (Garantir pureza de dominio, imports restritos a `adapters/`).
- [x] `PYTHONPATH=src python3 -m unittest discover -s tests -t .` (54 testes passando).
- [x] Teste mecanico: Executar tools via runtime real (`code_execution` e `http_request`) dentro do container `orchestrator`.
- [x] Teste mecanico: Validar geracao de surprise do Right Hemisphere em runtime. Resultado: backend `active_inference::numpy_vfe_fallback` observado apos reidratar episodios com `latent_vector` persistido.
- [x] Teste mecanico: Validar persistencia e reload de triples no grafo semântico local (`knowledge_graph.jsonl`) e injecao no `MemoryContext` via API.
- [x] Teste mecanico: Reiniciar containers, validar `/health`, `/ready`, `/v1/chat/completions`, `/v1/telemetry/dashboard/{session_id}` e correlacionar com logs de `orchestrator` e `qdrant`.

## Progress
- [x] Fase 1: Expansao do ToolRegistry.
- [x] Fase 2: Percepcao Avancada (pymdp).
- [x] Fase 3: Memoria Semantica Estruturada (nano-graphrag).
- [x] Fase 4: Aprendizado Continuo (DSPy).

## Decision Log
- 2026-03-29: Plano criado priorizando melhorias do nucleo cognitivo. Solucoes focadas no ambiente local.
- 2026-03-29: A ferramenta `browser_read` com Playwright foi classificada como prioridade menor para evitar inchaco de dependencias; foco no `code_execution` e `http_request`.
- 2026-03-29: `pymdp`, `nano-graphrag` e `DSPy` foram integrados como dependencias opcionais. O runtime agora usa os caminhos reais quando presentes e faz fallback explicito quando o ambiente local nao tem essas bibliotecas instaladas.
- 2026-03-29: O grafo semântico foi integrado sem retrabalho de extracao de entidades; o `SleepModeConsolidator` continua sendo a fonte das triples e o adapter novo cuida de indexacao, persistencia e busca multi-hop.
- 2026-03-29: A validacao operacional encontrou dois gaps de integracao e eles foram corrigidos na mesma sprint: o Qdrant precisava persistir `latent_vector` para o active inference funcionar entre processos, e o store de grafo precisava recarregar o arquivo quando o `sleep_mode` rodava em outro processo.
