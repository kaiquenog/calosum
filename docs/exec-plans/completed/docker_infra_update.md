# Atualização Docker Infra & Right Hemisphere

## Purpose
Tornar o Calosum 100% produtivo após as implementações da Fase 1/2. O agente servirá conexões de forma agnóstica de SO via Docker Compose através da porta 8000, com suporte a FastAPI WebServer embutido no container e persistência ligada à instância empacotada do Qdrant.

## Scope
- `deploy/Dockerfile`
- `deploy/docker-compose.yml`
- `docs/INFRASTRUCTURE.md`
- Avaliação de `domain/right_hemisphere.py`.

## Validation
Validação visual das configs e garantia mecânica que os arquivos modificados não afetam as diretrizes de pureza de imports (harness). O arquivo de Dockerfile garantirá instalação satisfatória do `pyproject.toml`.

## Progress
Planning Mode.

## Decision Log
- Decidido que `left-hemisphere` e `right-hemisphere` containers (com `sleep 3600`) serão removidos do Topology do docker, para não causarem conflito DNS `ConnectionRefused` quando a Engine disparar requisições ativas. O Qwen-9B será o maestro.
- O Right Hemisphere atual não exige refatoração pesada pois seu algorítmo sintético não gera impedimentos de CPU ou rede, cabendo adequadamente dentro da alocação de processo isolada.
