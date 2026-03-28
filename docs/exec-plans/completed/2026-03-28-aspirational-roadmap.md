# Implementation Plan: Aspirational Roadmap Fulfillment

## Purpose
Executar sistematicamente os passos restantes para levar o projeto do estado atual ao modelo aspiracional descrito em `INIT_PROJECT.MD` e `production-roadmap.md`, partindo de menor para maior complexidade.

## Scope
1. **[BAIXA COMPLEXIDADE]** Instalar dependências, validar o REPL interativo e implementar **Action Routing (Tools)** reais (`web_search` com DuckDuckGo, `file_write`) no `StrictLambdaRuntime`.
2. **[BAIXA COMPLEXIDADE]** Criar a **API REST / SSE** via FastAPI para expor o Orquestrador, permitindo conexões externas e testes vivos.
3. **[MÉDIA COMPLEXIDADE]** Implementar o **Sistema de Permissões** para proteger o ActionRuntime.
4. **[MÉDIA COMPLEXIDADE]** Evoluir o mecanismo de **Retentivas (Markov Chains)** no loop de correção metacognitiva.
5. **[ALTA COMPLEXIDADE]** Desenvolver o script de **Destilação Episódica ("Modo Noturno")** que resume dados brutos do Mem0/Qdrant.

## Validation
- Cada fase deve passar pelos testes locais (`python3 -m unittest`) e testes de harness (`python3 -m calosum.harness_checks`).
- O REPL e a API devem se comunicar perfeitamente com o adaptador Qwen.

## Progress
- [x] Criação do Plano.
- [x] Fase 1: Action Routing e Ferramentas Básicas (Em Andamento)
  - [x] `search_web` e `write_file` adicionados no `StrictLambdaRuntime`.
  - [x] Prompts do LLM atualizados para expor a lista de ferramentas dinamicamente.
- [x] Fase 2: API FastAPI
  - [x] Rota `/v1/chat/completions` (REST) implementada e validada.
  - [x] Rota `/v1/chat/sse` adicionada usando `sse-starlette` para Server-Sent Events.
- [x] Fase 3: Permissões e Segurança
  - [x] Criação do dicionário `vault` em `InfrastructureSettings` que lê de `CALOSUM_VAULT_*`.
  - [x] Injeção do `vault` no `ConcreteActionRuntime` via `CalosumAgentBuilder`.
- [x] Fase 4: Retentivas
  - [x] Padronização do status de erro no `ConcreteActionRuntime` para `status="rejected"` a fim de engatilhar a auto-correção via Markov Chain/HTN.
  - [x] Validação do loop de retry do `AgentExecutionEngine` onde falhas de ferramentas viram feedback de execução.
- [x] Fase 5: Destilação Episódica
  - [x] Adaptação do `QdrantDualMemoryAdapter` para invocar o `SleepModeConsolidator`.
  - [x] Extração de episódios do Qdrant e injeção como `SemanticRule`s no próprio Qdrant, consolidando o "Modo Noturno".
  
## Decision Log
- Começamos pelas ferramentas (Action Routing) e API REST pois o CLI Chat já teve um esboço inicial inserido recentemente no código (Fase 1 do Roadmap já estava quase terminada estruturalmente).
