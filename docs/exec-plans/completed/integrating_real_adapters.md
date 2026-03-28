# Integrating Real Adapters: Qwen3.5-9B, Qdrant & Real Actions

## Context
A análise profunda da arquitetura atual revelou que as interfaces do padrão `Ports and Adapters` estão robustas, mas os componentes que implementam as interfaces (`LeftHemisphereLogicalSLM`, `PersistentDualMemorySystem`, `StrictLambdaRuntime`) ainda funcionam primariamente como simuladores determinísticos sem real impacto externo ou IA embarcada. O objetivo deste plano é implementar adaptadores reais.

## Architectures & Boundaries
Seguindo os preceitos de `AGENTS.md` e `ARCHITECTURE.md`:
- Todos os novos códigos que conversam com I/O externo ficarão izolados no diretório **`src/calosum/adapters/`**.
- Os contratos fundamentais em `src/calosum/ports.py` permanecem imutáveis.
- A orquestração dentro de `orchestrator.py` permanecerá alheia às implementações reais, usando injeção de dependência via `factory.py`.

## Tasks

### 1. New Module: Adapters
- Criar a pasta e os arquivos.
- `src/calosum/adapters/llm_qwen.py`: Usa a API (compatível com OpenAI por ex, ou HTTPX client) para solicitar conclusões ao modelo `Qwen3.5-9B`, pedindo saídas em JSON que correspondam ao `LeftHemisphereResult`.
- `src/calosum/adapters/memory_qdrant.py`: Implementa cliente para o banco vetorial **Qdrant**, guardando embbedings e recuperando contexto real com as regras semânticas na memória.
- `src/calosum/adapters/action_runtime.py`: Extensão que realmente cumpre comandos caso a "Ação Primitiva" autorizada chegue nela (ex.: print no terminal, ou simulações realistas de fetch).

### 2. Dependencies
- Adicionar ao `pyproject.toml` as bibliotecas `httpx`, `qdrant-client` e `pydantic`.

### 3. Integrations (Factory)
- Em `factory.py`, instanciar essas classes e fornecê-las ao construtor do `CalosumAgent` dependendo de um perfil como `--infra-profile real`.

## Acceptance Criteria
- Todos os novos arquivos passam no `harness_checks.py`.
- Suíte `unittest` continua verde.
- O código do repositório fica consideravelmente mais limpo e organizado no novo padrão de projeto em subpastas (`calosum/adapters`).

## Purpose
Integrar adaptadores reais para as ports do agente.

## Scope
Left Hemisphere, Dual Memory, Action Runtime.

## Validation
Testes unitários e harness checks passando verdinhos.

## Progress
Finalizado.

## Decision Log
Decidimos usar a API local pseudo-OpenAI para ligar o Qwen-9B para gerar JSON via HTTPX.
O Qdrant será conectado através da SDK assíncrona.
