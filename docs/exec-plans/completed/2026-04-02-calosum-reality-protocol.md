Plano de Execução: Calosum Reality Protocol

    1 # 2026-04-02-calosum-reality-protocol
    2
    3 ## Purpose
    4 O objetivo deste plano é reconstruir o núcleo cognitivo do Calosum, substituindo implementações "mockadas" e heurísticas arbitrárias por mecânicas reais de **Active
      Inference** e **JEPA-Grounded World Modeling**. Atualmente, o framework possui uma infraestrutura sólida de software, mas falha na execução das promessas
      neuro-simbólicas SOTA.
    5
    6 ## Scope
    7 Este plano abrange quatro áreas críticas:
    8 1. **Matemática Cognitiva:** Implementação de Variational e Expected Free Energy (VFE/EFE).
    9 2. **Infraestrutura Latente:** Migração do IPC JSON para Apache Arrow e integração real com pesos V-JEPA 2.
   10 3. **Ponte Neuro-Simbólica:** Implementação de Soft Prompting e Grounding Semântico.
   11 4. **Evolução Multiagente (GEA):** Ativação de branching cognitivo real via batching de inferência.
   12
   13 ### Sprint 1: Foundation of Truth (Semanas 1-3)
   14 - Implementar `src/calosum/shared/utils/math_cognitive.py` com suporte a tensores.
   15 - Refatorar `InputPerceptionState` para incluir parâmetros de distribuição (mu, logvar).
   16 - Substituir IPC do `jepa-rs` por Arrow ou Memory-Map.
   17
   18 ### Sprint 2: World Modeling (Semanas 4-6)
   19 - Ativar o loop de predição latente action-conditioned no `VJepa21RightHemisphereAdapter`.
   20 - Criar o `SoftPromptProjector` para alinhamento JEPA -> LLM.
   21
   22 ### Sprint 3: Recursive Reasoning (Semanas 7-9)
   23 - Implementar o mecanismo de backtracking real no `RlmLeftHemisphereAdapter`.
   24 - Integrar o `StrictLambdaRuntime` com feedback de erro para o World Model.
   25
   26 ### Sprint 4: Collective Evolution (Semanas 10-12)
   27 - Ativar o `GEAReflectionController` com seleção baseada em EFE.
   28 - Implementar o ciclo de `Sleep Mode` com otimização DSPy real.
   29
   30 ## Validation
   31 A validação será contínua via:
   32 - **Testes de Fidelidade Cognitiva:** Correlação entre erro de predição e `surprise_score` > 0.85.
   33 - **Benchmarks de Performance:** Latência inter-processos < 1ms para dados latentes.
   34 - **Harness Checks:** Garantia de 0 violações de importação em `src/calosum/harness_checks.py`.
   35 - **E2E Integration:** Teste de "Novelty Detection" onde o agente deve identificar mudanças externas imprevistas no ambiente de execução.
   36
   37 ## Progress
   38 - [x] Sprint 1: Foundation of Truth
   39     - [x] Math Cognitive Utils
   40     - [x] Arrow/IPC Refactor
   41     - [x] State Schema Update
   42 - [x] Sprint 2: World Modeling
   43 - [x] Sprint 3: Recursive Reasoning
   44 - [x] Sprint 4: Collective Evolution
   45
   46 ## Decision Log
   47 - **2026-04-02:** Decidido abandonar o fallback de MLP randômica em favor de erros explícitos de carregamento de modelo para garantir integridade científica.
   48 - **2026-04-02:** Optado por Apache Arrow para IPC devido à necessidade de transportar tensores de alta dimensionalidade entre Rust e Python sem overhead de
      serialização.

  Este documento servirá como a "fonte da verdade" para a evolução do Calosum nos próximos meses, garantindo que cada commit contribua para a maturidade real do sistema.
