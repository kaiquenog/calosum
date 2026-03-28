# Fases 1 e 2: Interface Real e Ações Físicas

## Purpose
Tornar o repositório utilizável interativamente após meses focado na "arquitetura de backroom". Prover o Loop infinito REPL no CLI. Ligar a fundação do Orquestrador a uma API Http em FastAPI. Ensinar os motores locais a efetuarem buscas na web nativa com DDG.

## Scope
- Pacote: `bootstrap` (`api.py`, `settings.py`, `cli.py`)
- Pacote: `adapters` (`action_runtime.py`)
- Testes: CLI tests, Unit tests básicos da API se requerido.

## Validation
O `harness_checks.py` tem que ser capaz de auditar a nova rota HTTP `api.py` como pertecente apenas ao `bootstrap` (isto é, FastAPI não entra na camada de Domínio, servindo apenas de wrapper HTTP/JSON). Unit tests passarão confirmando integridade estrutural.

## Progress
Planning Mode. Elaborado o `implementation_plan.md` listando ferramentas (uvicorn, fastapi, duckduckgo-search) que serão injetadas via `pyproject.toml`.

## Decision Log
- Decidimos via metodologia Harness não espalhar código HTTP no Domínio. A API residirá exclusivamente na mesma camada que o `factory.py` (a do Bootstrap), isolando as restrições web.
- O Web Search adotou `duckduckgo-search` livre, e não API SERP de mercado, para otimizar barreiras de entrada financeiras do projeto nativo.
