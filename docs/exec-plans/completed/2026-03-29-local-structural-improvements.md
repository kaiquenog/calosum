# Execution Plan: Melhorias Estruturais Locais (Baseado em Analise de Maturidade)

## Purpose
O objetivo deste plano é aplicar as melhorias estruturais locais sugeridas pelo `2026-03-29-maturity-analysis-report.md`. Conforme diretriz, as melhorias relacionadas à infraestrutura de produção (CI/CD, JWT, etc.) foram desconsideradas. O foco é manter a base de código limpa, madura, com boa governança local e preparada para o futuro.

## Scope
**Sprint 1: Saúde do Código & API**
- Remoção de código morto (`advanced_interfaces.py`).
- Remoção da referência do `advanced_interfaces.py` no `harness_checks.py`.
- Adição de endpoints de `/health` e `/ready` na API do FastAPI.
- Habilitação explícita do Swagger UI no FastAPI (`docs_url="/docs"`).

**Sprint 2: Refinamento de Arquitetura & Domínio**
- Implementação de shutdown gracioso no `EventBus` (`stop()` method com cancellation).
- (Opcional/Avaliável) Adição de validações básicas (ex: Pydantic ou assert/property) em dataclasses críticas.
- (Opcional/Avaliável) Extração de heurísticas de provider para `provider_detection.py`.

**Sprint 3: Resiliência & Telemetria**
- Adição de retry com exponential backoff (via `tenacity`) nas chamadas HTTP (`httpx` e Qdrant/LLM).
- Implementação de Circuit Breaker.
- Otimização do `telemetry` para escrita em batch (se viável sem overengineering).

*Nota:* Testes serão atualizados localmente conforme as mudanças são feitas.

## Validation
- `python3 -m calosum.harness_checks` executando com sucesso (sem drift arquitetural).
- `python3 -m unittest discover -s tests -t .` executando com sucesso.
- Inicialização local da API (`python3 -m calosum.bootstrap.api`) validando o endpoint de health.

## Progress
- [ ] Criado plano de execução.
- [ ] Sprint 1 implementada.
- [ ] Sprint 2 implementada.
- [ ] Sprint 3 implementada.

## Decision Log
- **2026-03-29**: Optado por não incluir CI/CD, deploy, nem JWT, focando puramente no aspecto estrutural e local de maturidade do framework em Python.
