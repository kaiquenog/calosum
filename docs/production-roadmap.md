# Calosum Production Roadmap

Este documento lista os próximos passos estratégicos para evoluir o Calosum, ordenados rigorosamente por prioridade de impacto, considerando que a fundação estrita (Ports & Adapters) e as conexões básicas (Qwen + Qdrant) já estão em vigor.

## Fase 1: Interface e Usabilidade Básica (Prioridade Alta)

Atualmente o agente só interage via comandos CLI diretos (`run-turn` e `run-scenario`). Para experimentação contínua, precisamos de interfaces vivas.
1. **Interactive REPL (CLI Chat):** Criar um comando `python3 -m calosum.bootstrap.cli chat` que mantenha a sessão aberta no terminal para conversas interativas fluídas.
2. **API REST / SSE:** Criar uma camada de rede simples (ex: FastAPI) acoplada na camada `bootstrap` para servir a porta do `Orchestrator` via WebSockets ou Server-Sent Events (SSE).
3. **Integração Externa:** Desenvolver um script client conectando essa API ao Telegram ou WhatsApp para testes de campo.

## Fase 2: Robustez nas Ferramentas (Prioridade Média-Alta)

A estrutura do `ConcreteActionRuntime` já existe, mas suas "primitivas de ação" ainda não causam impacto real de rede ou disco.
1. **Routing de Ações (Tools):** Mapear assinaturas do Qwen para ferramentas atômicas reais (ex. `execute_web_search()` com DuckDuckGo, `execute_file_write()`).
2. **Sistema de Permissões:** O `ActionRuntime` precisará de um cofre de chaves (gerenciado via `settings.py`) para consumir APIs externas sem vazar contexto interno.
3. **Observabilidade (Agent UI):** Ligar os eventos que já caem no `OTLPJsonlTelemetrySink` em um dashboard como Jaeger ou Langfuse, para enxergar visualmente quando o Qwen falhou no formato.

## Fase 3: Raciocínio Avançado e Correção (Prioridade Média)

O Orquestrador suporta `CognitiveVariantSpec`, mas precisamos de controle dinâmico dessas rotas.
1. **Cadeias de Markov e Múltiplos Passos (HTN):** Expandir as retentivas de raciocínio. Se uma tool falha, injetar a mensagem de erro no prompt e re-pedir a ação.
2. **Active Inference (Free Energy):** Modificar o Hemisfério Direito para calcular a *Loss* preditiva do input do usuário. Entradas muito "surpreendentes" abaixam a temperatura do LLM e disparam mais passos de reflexão antes de agir.

## Fase 4: SOTA - Memória Dinâmica (Prioridade Baixa/Pesquisa)

1. **Destilação Episódica (Neuroplasticidade):** Criar o "Modo Noturno". Scripts que varrem o Qdrant de madrugada, consolidam o que foi aprendido, e treinam pequenos adaptadores LoRA. No dia seguinte, o Qwen-9B carrega os novos pesos, transformando *memória de trabalho* em *intuição implícita* (pesos do modelo).
