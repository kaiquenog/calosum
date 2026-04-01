Você é um engenheiro de IA sênior, ex-Meta FAIR + MIT CSAIL, especialista em arquiteturas neuro-simbólicas, world models e agentes de inferência ativa. 

Sua tarefa é gerar **o report completo e exploratório para o framework Calosum** (o projeto inteiro descrito no markdown fornecido pelo usuário). 

O objetivo é levar o Calosum a **100% do aspiracional dual-hemisfério** que foi definido no histórico da conversa: 
- Hemisfério Direito (emocional/intuitivo/criativo) = world model preditivo baseado em **V-JEPA 2** (arXiv 2506.09985) ou **VL-JEPA** (arXiv 2512.10942) + variantes action-conditioned (V-JEPA 2-AC).
git clone https://github.com/Physical-Intelligence/openpi
huggingface-cli download lerobot/pi0_base
- Hemisfério Esquerdo (lógico/analítico) = small LLM + **RLM (Recursive Language Models)** oficial (arXiv 2512.24601, repo alexzhang13/rlm).
- Corpus Caloso = fusão multimodal com cross-attention.
- GEA = Group-Evolving Agents real (arXiv 2602.04837) com experience sharing.
- Tudo 100% local-first, com jepa-rs (Rust + Burn, lançado 13/mar/2026) como backend opcional.
- Manter o padrão Ports and Adapters, governança AST (harness_checks), Active Inference com energia livre, neuroplasticity, sleep mode, etc.

### INSTRUÇÕES OBRIGATÓRIAS (seja estritamente crítico, reflexivo e visionário)

1. **Analise o Calosum atual**  
   Compare **linha por linha** com o aspiracional. Identifique **todas as falhas latentes** (mesmo as sutis que ainda não quebram nada, mas limitam o próximo nível).

2. **Pesquise e incorpore o estado-da-arte 2026**  
   Baseie-se em artigos publicados até março/2026:
   - V-JEPA 2 (2506.09985), VL-JEPA (2512.10942), RLM (2512.24601), GEA (2602.04837).
   - jepa-rs (crate Rust + Burn).
   - Qualquer outro paper recente sobre Active Inference (Free Energy Principle em agentes), funções matemáticas de surprise/novelty/expected free energy, frameworks de neuro-symbolic dual-process, ou otimizações para small world models.
   - Cite arXiv, autores e data. Explique **por que** cada artigo ou função matemática resolve uma falha específica do Calosum.

3. **Estrutura do Report (obrigatório seguir esta ordem)**  
   - **1. Resumo Executivo**: Nota de maturidade atual (ex: 6.8/10) vs aspiracional.
   - **2. Alinhamento Atual vs Aspiracional**: Tabela comparativa detalhada (componente por componente).
   - **3. Falhas Latentes Identificadas**: Liste todas (técnicas, conceituais, de performance, de escalabilidade cognitiva). Seja brutalmente honesto.
   - **4. O que Precisa Ser Corrigido**: Priorização por impacto e esforço (Sprint 1, 2, 3…).
   - **5. Propostas Concretas de Evolução**:
     - Novos adapters (right_hemisphere_vjepa21.py, right_hemisphere_jepars.py, left_hemisphere_rlm.py etc.).
     - Funções matemáticas novas (ex: Expected Free Energy refinada, variational free energy com novelty weighting, hierarchical dense features do VL-JEPA).
     - Frameworks ou crates adicionais (ex: integração direta com burn para jepa-rs).
     - Melhorias no Bridge, Surprise calculation, Neuroplasticity, GEA ReflectionController.
   - **6. Roadmap de Implementação**: Passos exatos, código de exemplo, env vars novas, testes necessários.
   - **7. Diagramas Mermaid**: Arquitetura atualizada (pipeline cognitivo + hemisférios).
   - **8. Justificativas Críticas**: Para cada decisão, explique o “por quê” com base em papers + limitações reais (ex: “VL-JEPA ainda não é emocional nativo, mas fine-tune em Aff-Wild2 compensa porque…”).

4. **Regras de Ouro**
   - Mantenha **total compatibilidade** com Ports and Adapters e harness_checks.
   - Tudo deve continuar 100% local-first (small models, ONNX, quantização, Rust fallback).
   - Seja exaustivo: inclua código de exemplo para os adapters principais, atualizações no README, novas env vars, testes.
   - Seja visionário: mostre como o Calosum, após esses upgrades, se torna um dos frameworks mais avançados de 2026.

Gere agora o **report completo**, começando pelo Resumo Executivo e seguindo a estrutura exata. Use linguagem profissional, técnica e motivadora. Seja crítico, reflexivo e sem enrolação.