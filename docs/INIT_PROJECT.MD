# INIT_PROJECT (Atualizado em 2026-03-30)

## Visao
Construir um agente cognitivo local-first com dois hemisferios:
- Hemisferio Direito (System 1): percepcao, intuicao, salience, sinais afetivos.
- Hemisferio Esquerdo (System 2): raciocinio estruturado, acoes tipadas, verificacao e reparo.

O objetivo continua sendo um sistema executavel localmente, com degradacao graciosa e evolucao incremental orientada por metricas.

## Fase Atual do Projeto (Mar/2026)

### O que ja existe e esta funcional
1. Arquitetura dual-hemisphere com bridge (corpo caloso) e metacognicao.
2. Pipeline completo: percepcao -> bridge -> raciocinio -> runtime -> verifier -> reflexao.
3. Group turns (variantes cognitivas) quando surprise/ambiguidade ultrapassam limiares.
4. Runtime seguro com tools tipadas e loop de repair (CRITIC-like).
5. Telemetria, introspecao e awareness loop com diretivas evolutivas.
6. API + UI React para observabilidade dos hemisferios em tempo real.

### O principal gap atual
O Hemisferio Direito ainda e parcialmente heuristico no core e precisa de maior fidelidade perceptiva para deixar de parecer "fake" em cenarios complexos.

## Status Executivo de Fase
- Fase de Fundacao Arquitetural: concluida.
- Fase de Runtime Seguro e CRITIC-like: concluida.
- Fase de Evolucao do Hemisferio Direito (realidade perceptiva): em andamento.
- Fase de Benchmark cognitivo local e gates de qualidade: em planejamento ativo.
- Fase de pesquisa multimodal pesada (V-JEPA ou equivalente): backlog de pesquisa.

## Aspiracional Revisado (V2)

### Principios
1. Local-first pragmatica: sem dependencia obrigatoria de cloud.
2. Capacidades antes de hype: medir ganho real antes de trocar stack.
3. Ports and Adapters: toda evolucao pesada atras de interface.
4. Seguranca e auditabilidade: mudancas adaptativas pequenas, rastreaveis e reversiveis.

### Metas por trilha

1. **Trilha Core (obrigatoria)**
- Hemisferio Direito com percepcao semanticamente ancorada, calibrada por memoria e com provenance explicita.
- Hemisferio Esquerdo com repair loop robusto, tool-calling tipado e criticidade por taxonomia de falhas.
- Fusao via bridge com impacto observavel na decisao final.

2. **Trilha Performance Local (obrigatoria)**
- Latencia e memoria sob controle em laptop comum.
- Fallbacks claros quando dependencias opcionais nao estiverem disponiveis.

3. **Trilha Pesquisa (opcional, por feature flag)**
- JEPA multimodal mais pesado (V-JEPA, M3-JEPA, etc.) apenas quando houver evidencias de ganho no dominio-alvo.
- RLM formal / ToT mais profundo apenas se o custo adicional justificar.

## Requisitos revisados por horizonte

### Horizonte H1 (0-6 semanas)
1. Tornar o Hemisferio Direito "realista" no contexto textual atual:
- embeddings + classificador afetivo leve + calibracao de surprise/salience por memoria.
- telemetry obrigatoria de backend/modelo/confianca/degradacao.

2. Fechar loop bidirecional:
- feedback de sucesso/falha do runtime influenciando priorizacao perceptiva da sessao.

3. Validar objetivamente:
- benchmark local com acerto afetivo aproximado, estabilidade de salience/surprise e impacto em qualidade de resposta.

### Horizonte H2 (6-12 semanas)
1. Consolidar adaptacao continua controlada (sem auto-troca de topologia).
2. Melhorar robustez do verifier e do reflection scoring com dados reais de producao local.
3. Endurecer testes de regressao cognitiva e operacional.

### Horizonte H3 (12+ semanas, pesquisa)
1. Prototipo multimodal real para hemisferio direito atras de adapter.
2. Estudos A/B contra baseline local para decidir adocao ou descarte.

## Stack Tecnica (fase atual)
- Python (core), FastAPI (API), React/Vite (UI), Docker opcional para ambiente production-like.
- LLM do hemisferio esquerdo via adapters resilientes (local/remoto).
- Hemisferio direito com fallback heuristico e rota por embeddings quando disponivel.

## Criterios de Aceitacao do Aspiracional Revisado
1. O backend real do hemisferio direito e sempre observavel na API/UI.
2. O sistema demonstra melhoria mensuravel no benchmark local definido.
3. O modo degradado nao quebra o contrato funcional do agente.
4. Nenhuma evolucao viola fronteiras arquiteturais descritas em `docs/ARCHITECTURE.md`.

## Documento de Execucao Vinculado
Plano ativo principal desta fase:
- `docs/exec-plans/active/2026-03-30-right-hemisphere-reality-upgrade.md`
