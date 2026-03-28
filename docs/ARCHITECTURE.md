# Architecture

## Objetivo

Manter o crescimento do projeto legivel para humanos e agentes, com fronteiras pequenas e verificaveis.

## Padrao Aplicado

O projeto usa `Ports and Adapters` para fronteiras e `Builder/Abstract Factory` para compor infraestrutura.

- Em `shared/ports.py` residem nossos contratos de interfaces estáveis.
- Em `bootstrap/settings.py` residem os perfis flexíveis de execução local (incluindo o Vault de credenciais).
- Em `bootstrap/factory.py` é configurado o bootstrap do agente, injetando `adapters` no `domain`.

## Camadas

1. **`shared/`** (`types.py`, `ports.py`, `async_utils.py`, `serialization.py`)
   Tipos compartilhados, contratos de dados e utilitários puros de serialização.
2. **`domain/`** (`bridge.py`, `orchestrator.py`, `right_hemisphere.py`, `left_hemisphere.py`, `memory.py`, `persistent_memory.py`, `runtime.py`, `telemetry.py`, `metacognition.py`, `agent_execution.py`)
   Modelos de negócios do agente neuro-simbólico. Pura lógica sem detalhes I/O diretos.
3. **`adapters/`** (`llm_qwen.py`, `memory_qdrant.py`, `action_runtime.py`, `right_hemisphere_hf.py`, `night_trainer.py`)
   Implementações concretas dos ports que conversam com LLMs reais, bancos vetoriais, redes neurais de embedding ou executam tarefas lógicas seguras.
4. **`bootstrap/`** (`settings.py`, `factory.py`, `cli.py`, `api.py`, `__main__.py`)
   Entrada da aplicação que avalia configurações locais, inicia a API REST/SSE e monta todo o motor ligando os mundos.

## Interface de Usuário (UI)

O projeto também possui um componente frontend na pasta `ui/` construído com React, Vite e Tailwind. Este painel consome as rotas expostas em `bootstrap/api.py` para exibir a telemetria separada por hemisférios em tempo real.

## Regras

- Pacote `shared` não depende de outros pacotes internos. Serve como base de comunicação de dicionários, data classes e portas (`Protocols`).
- Pacote `domain` define o core. Ele NUNCA deve tentar importar bibliotecas SDK de "adapters" nem as instâncias do "bootstrap".
- Pacote `adapters` obedece cegamente a interface em `shared`. Não toma decisões fora de traduzir a infra.
- Pacote `bootstrap` é o único capaz e autorizado a instanciar `adapters` concretos injetando-os nas instâncias do `domain` de acordo com configs do painel `settings.py`.
- O agente tem uma entrada limpa e orquestrada pelo `orchestrator.py`. Interações isoladas da `cli.py` ou `api.py` não vazam contexto pro `domain`.

## Crescimento Controlado

- Toda nova integracao externa deve entrar em `adapters` atrás de um `Protocol`.
- A arquitetura está mecanicamente enforçada pelo `harness_checks.py`. Exceções devem ser evitadas para poupar crescimento desordenado.
- Toda mudanca cross-cutting deve ter plano versionado.
