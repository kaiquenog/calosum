# Infrastructure

## Padrao de Organizacao

Infraestrutura e bootstrap usam `Builder/Abstract Factory`.

- `settings.py` resolve perfil e endpoints
- `factory.py` escolhe adapters concretos
- `docker-compose.yml` define o ambiente local de infraestrutura interligada

## Perfis

- `ephemeral`: tudo em memoria
- `persistent`: persistencia local em disco
- `docker`: persistencia em volume, tracing via collector, vector DB disponivel na rede do compose e Web Server ativo na porta 8000.

## Compose Atual

- `orchestrator`: processo principal isolado que roda a API Rest/SSE (FastAPI) em `http://localhost:8000`.
- `qdrant`: banco Vetorial que atua com o adaptador Dual Memory.
- `otel-collector`: recepcao de OTLP.
- `jaeger`: visualizacao UI de traces.

Observacao:
- Containers fakes criados na fundação da arquitetura (`right-hemisphere` / `left-hemisphere`) foram extintos. O agente usa abstrações ativas do pacote `adapters` que interagem com o mundo real ou APIs externas, mantendo as regras estritas da aplicação rodando inteiramente na V-NET do container principal.

## Pendencias Ja Encaixadas Na Infra

- `qdrant` está 100% vitalizado por `adapters.memory_qdrant`.
- O servidor interativo pode ser contatado via API local ou com a CLI `python3 -m calosum.bootstrap.cli chat`.
