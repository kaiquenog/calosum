# Next Gap Closure

## Purpose

Fechar a proxima faixa de gaps estruturais do Calosum depois da rodada de guard-rails: transformar o runtime em executor semantico limitado, fortalecer a memoria vetorial com embeddings reais ou configuraveis e ampliar a adaptacao persistente do bridge para alem de limiares estaticos.

## Scope

- evoluir `StrictLambdaRuntime` de "semantic guard" para interpretador restrito da DSL atual
- introduzir backend de embeddings configuravel para a memoria Qdrant, com fallback explicito
- reforcar a neuroplasticidade do bridge com estado calibrado e historico persistente
- atualizar ou adicionar testes de regressao
- atualizar documentacao operacional se a configuracao mudar

## Validation

- `PYTHONPATH=src python3 -m calosum.harness_checks`
- `PYTHONPATH=src python3 -m unittest discover -s tests -t .`
- `npm run lint`
- `npm run build`

## Progress

- iniciado em 2026-03-28
- concluido em 2026-03-28
- runtime lambda passou a executar uma DSL restrita de verdade, com ordem de execucao derivada do programa simbolico
- hemisferio esquerdo local passou a emitir uma sequencia simbolica compatível com o runtime semantico
- adapter Qdrant ganhou backend de embeddings configuravel com derivacao automatica para OpenAI e fallback lexical explicito
- bridge passou a persistir calibracao de saliencia/temperatura e historico de reflexao, nao apenas limiares basicos
- testes adicionados para ordem/condicional do runtime, backend HTTP de embeddings e derivacao de embeddings no builder
- sanitizacao completa executada ao final

## Decision Log

- decidido extrair a DSL do runtime para `domain/runtime_dsl.py` em vez de relaxar o limite de tamanho do harness
- decidido manter o embedding lexical como fallback operacional, mas introduzir `text_embeddings.py` para backends reais sem acoplar o dominio a SDKs externos
- decidido usar calibracao persistente do bridge e historico de reflexao local como passo intermediario antes de um loop completo de treinamento de pesos
