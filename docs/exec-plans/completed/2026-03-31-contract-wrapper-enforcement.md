# Contract Wrapper Enforcement For Left/Right Hemispheres

## Purpose
Padronizar o enforcement de contrato de saída para todos os adapters do hemisfério esquerdo (LLM/SLM) e também para o lado direito (percepção), reduzindo variação de comportamento entre backends.

## Scope
- Criar wrappers de contrato para `LeftHemispherePort` e `RightHemispherePort`.
- Integrar wrappers no `bootstrap/backend_resolvers.py` para cobrir todos os adapters atuais.
- Garantir compatibilidade com failover/retry existentes.
- Adicionar testes unitários do enforcement.
- Atualizar regras do harness para o novo módulo.

## Validation
- Testes focados: `PYTHONPATH=src .venv/bin/python -m unittest tests.test_contract_wrappers tests.test_factory tests.test_factory_backends_2026 tests.test_llm_adapter`.
- Harness: `PYTHONPATH=src python3 -m calosum.harness_checks`.
- Suite completa: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t .`.
- Verificação runtime em container (`rlm` e `qwen`) com flags de wrapper nas telemetrias esquerda/direita.

## Progress
- [x] Abrir plano ativo.
- [x] Implementar wrappers de contrato.
- [x] Integrar wrappers no resolver.
- [x] Criar/ajustar testes.
- [x] Executar validação completa.

## Decision Log
- 2026-03-31: Estratégia escolhida: wrapper único por hemisfério com normalização mínima e telemetria explícita de ajustes, sem alterar contratos públicos dos ports.
- 2026-03-31: Wrapper do hemisfério direito foi aplicado sobre o adapter base (antes do Active Inference), preservando `ActiveInferenceRightHemisphereAdapter` como camada externa.
- 2026-03-31: `harness_checks` e `unittest discover` seguem falhando por artefato obrigatório ausente já existente (`docs/product-specs/calosum-system.md`), não introduzido por esta mudança.
