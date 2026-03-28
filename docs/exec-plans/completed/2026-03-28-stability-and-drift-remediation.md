# Stability And Drift Remediation

## Purpose

Reduzir o gap entre a arquitetura declarada e o estado executavel real do Calosum, estabilizando bootstrap, empacotamento, docs e UI para crescimento mais previsivel.

## Scope

- desacoplar o bootstrap de dependencias pesadas obrigatorias
- corrigir entrypoints e referencias quebradas de documentacao
- fortalecer fallbacks e tipos em pontos onde o runtime atual aceita estados parciais
- alinhar UI e verificacoes basicas com o contrato real de telemetria
- adicionar ou ajustar testes para cobrir os regressos encontrados

## Validation

- `PYTHONPATH=src python3 -m calosum.harness_checks`
- `PYTHONPATH=src python3 -m unittest discover -s tests -t .`
- `npm run lint`
- `npm run build`

## Progress

- concluido em 2026-03-28
- iteracao 1 entregou fallback explicito no bootstrap, lazy init da API e correção do entrypoint empacotado
- iteracao 2 alinhou hidratação de memória persistida, README/docs e UI ao contrato real de telemetria

## Decision Log

- manter o padrao ports and adapters, mas preferir fallback explicito a falha dura quando a dependencia pesada for opcional
- tratar drift de docs e empacotamento como bug funcional, nao apenas melhoria editorial
- hidratar episodios legados com placeholders tipados foi escolhido em vez de relaxar o contrato global de `MemoryEpisode`
- a UI passou a consumir base URL configuravel e exibir `execution` e `reflection` para bater com o dashboard real
