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
2. **`domain/`** (`advanced_interfaces.py`, `bridge.py`, `orchestrator.py`, `right_hemisphere.py`, `left_hemisphere.py`, `memory.py`, `persistent_memory.py`, `runtime.py`, `telemetry.py`, `metacognition.py`, `agent_execution.py`)
   Modelos de negócios do agente neuro-simbólico. Pura lógica sem detalhes I/O diretos.
3. **`adapters/`** (`llm_qwen.py`, `memory_qdrant.py`, `action_runtime.py`, `right_hemisphere_hf.py`, `night_trainer.py`)
   Implementações concretas dos ports que conversam com LLMs reais, bancos vetoriais, redes neurais de embedding ou executam tarefas lógicas seguras.
4. **`bootstrap/`** (`settings.py`, `factory.py`, `cli.py`, `api.py`, `__main__.py`)
   Entrada da aplicação que avalia configurações locais, inicia a API REST/SSE e monta todo o motor ligando os mundos.

## Governanca de Harness

Fora das quatro camadas principais, o repositorio mantem `harness_checks.py` na raiz do pacote `calosum` como utilitario de governanca. Ele nao faz parte do runtime do agente; sua funcao e validar artefatos obrigatorios, planos, limites de modulo e fronteiras de importacao.

## Interface de Usuário (UI)

O projeto também possui um componente frontend na pasta `ui/` construído com React, Vite e Tailwind. Este painel consome as rotas expostas em `bootstrap/api.py` para exibir a telemetria separada por hemisférios, execução e reflexão.

## Regras

- Pacote `shared` não depende de outros pacotes internos. Serve como base de comunicação de dicionários, data classes e portas (`Protocols`).
- Pacote `domain` define o core. Ele NUNCA deve tentar importar bibliotecas SDK de "adapters" nem as instâncias do "bootstrap".
- Pacote `adapters` obedece cegamente a interface em `shared`. Não toma decisões fora de traduzir a infra.
- Pacote `bootstrap` é o único capaz e autorizado a instanciar `adapters` concretos injetando-os nas instâncias do `domain` de acordo com configs do painel `settings.py`.
- Quando um adapter opcional de infraestrutura local não estiver disponível, o `bootstrap` deve preferir fallback explícito a falha dura sempre que isso não quebrar o contrato funcional.
- O agente tem uma entrada limpa e orquestrada pelo `orchestrator.py`. Interações isoladas da `cli.py` ou `api.py` não vazam contexto pro `domain`.

## Crescimento Controlado

- Toda nova integracao externa deve entrar em `adapters` atrás de um `Protocol`.
- A arquitetura esta mecanicamente enforcada pelo `harness_checks.py` no que diz respeito a docs obrigatorios, planos, tamanho de modulo e fronteiras de importacao. Regras novas devem virar checks mecanicos quando se tornarem recorrentes.
- Toda mudanca cross-cutting deve ter plano versionado.
