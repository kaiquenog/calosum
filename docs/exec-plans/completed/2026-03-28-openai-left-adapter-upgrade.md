# OpenAI Left Adapter Upgrade

## Purpose

Atualizar a integracao do hemisferio esquerdo para suportar OpenAI de forma nativa e atual, reduzindo o gap entre o adapter legado orientado a Qwen/Ollama e os modelos atuais recomendados pela OpenAI.

## Scope

- evoluir o adapter do hemisferio esquerdo para suportar OpenAI Responses e Structured Outputs
- manter compatibilidade com endpoints OpenAI-compatible locais
- expor configuracao minima para raciocinio e provider
- validar se o `.env` atual esta pronto para OpenAI e corrigir o codigo para esse fluxo
- adicionar testes e atualizar documentacao operacional

## Validation

- `PYTHONPATH=src python3 -m calosum.harness_checks`
- `PYTHONPATH=src python3 -m unittest discover -s tests -t .`
- `npm run lint`
- `npm run build`

## Progress

- iniciado em 2026-03-28
- concluido em 2026-03-28
- adapter do hemisferio esquerdo atualizado para autodetectar OpenAI oficial e usar Responses API com Structured Outputs
- mantida compatibilidade com endpoints locais OpenAI-compatible via chat completions
- bootstrap passou a descrever corretamente backend, provider e reasoning effort do hemisferio esquerdo
- adicionados testes de regressao para OpenAI Responses e endpoint compativel local
- documentacao operacional atualizada para o fluxo OpenAI

## Decision Log

- decidido manter o arquivo `llm_qwen.py` por compatibilidade, mas evolui-lo para adapter generico em vez de criar uma troca abrupta de nomenclatura
- decidido autodetectar `https://api.openai.com/v1` como OpenAI oficial e preferir Responses API, alinhando o projeto com a recomendacao atual da OpenAI para novos projetos
- decidido normalizar aliases comuns de modelo (`gpt-5.4-mini` -> `gpt-5-mini`) para reduzir erro operacional no `.env`
