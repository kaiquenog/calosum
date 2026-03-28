# Gap Remediation By Complexity

## Purpose

Reduzir os gaps prioritarios restantes do Calosum em ordem de complexidade crescente, preservando a arquitetura atual e sem expandir o hemisferio direito para multimodal real nesta rodada.

## Scope

- tornar o runtime lambda semanticamente mais verificavel
- substituir vetores dummy e `scroll` generico do adapter Qdrant por indexacao e busca vetorial reais
- ampliar a neuroplasticidade da reflexao para ajustar mais do que um unico limiar
- adicionar ou atualizar testes de regressao
- atualizar documentacao se o comportamento operacional mudar

## Validation

- `PYTHONPATH=src python3 -m calosum.harness_checks`
- `PYTHONPATH=src python3 -m unittest discover -s tests -t .`
- `npm run lint`
- `npm run build`

## Progress

- iniciado em 2026-03-28
- concluido em 2026-03-28
- runtime lambda passou a validar alinhamento semantico entre programa e fronteira de acoes
- adapter Qdrant deixou de usar vetores dummy e passou a indexar e consultar vetores deterministas por similaridade
- reflexao passou a ajustar e persistir mais parametros do corpo caloso, nao apenas `salience_threshold`
- testes de regressao adicionados para runtime, qdrant e persistencia de neuroplasticidade
- sanitizacao completa executada ao final

## Decision Log

- decidido manter a evolucao do runtime em modo "semantic guard" em vez de um interpretador lambda completo nesta rodada, para reduzir risco sem quebrar contratos existentes
- decidido usar embeddings lexicais deterministas no Qdrant como passo intermediario entre vetores dummy e um embedder neural dedicado, preservando consistencia entre indexacao e consulta sem acoplar mais dependencias pesadas
- decidido persistir a neuroplasticidade em arquivo de configuracao do tokenizer para carregar adaptacoes entre processos sem introduzir um backend adicional
