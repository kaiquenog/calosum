# Calosum V2: The Aspirational Dual-Hemisphere Evolution

## Purpose
This plan formalizes the transition of Calosum from a "Heuristic-Simulated" dual-process architecture to a "Unified Latent World Model" (ULWM) based on Active Inference principles. The goal is to reach 100% of the aspirational vision: perception as predictive modeling (JEPA), reasoning as grounded logic (LTN + DSPy), and decision-making as uncertainty reduction (Expected Free Energy).

## Scope
- **Domain**: `right_hemisphere.py`, `left_hemisphere.py`, `orchestrator.py`, `memory.py`, `evolution.py`.
- **Adapters**: New implementations for `JEPA` (Vision/Text), `DSPy` (Prompt Distillation), and `LTN` (Differentiable Logic).
- **Harness**: Extension of `harness_checks.py` to verify "Architectural Entropy" and module boundary adherence.

## Validation
- **Mechanical Integrity**: `PYTHONPATH=src python3 -m calosum.harness_checks` must pass with 0 issues.
- **Unit Stability**: `PYTHONPATH=src python3 -m unittest discover -s tests -t .` - zero regressions.
- **Cognitive Fidelity**: Verify that `surprise_score` is now driven by KL-Divergence and branching by Expected Free Energy (EFE).

## Progress

### Sprint 1: Foundations of Perception (JEPA & KL-Divergence)
- [x] Create `src/calosum/adapters/right_hemisphere_jepa.py`.
- [x] Replace heuristic seed logic in `RightHemisphereJEPA` domain with latent encoder logic.
- [x] Refactor `_calculate_surprise` to use KL-Divergence $D_{KL}[Q(s|o) || P(s)]$.
- [x] Update `harness_checks.py` rules for new adapter module.

### Sprint 2: Cognition and Grounding (LTN & DSPy)
- [x] Implement `src/calosum/domain/differentiable_logic.py` (Logic Tensor Networks core).
- [x] Update `LeftHemisphereLogicalSLM` to apply logical constraints from memory during reasoning.
- [x] Integrate DSPy as an adapter to enable proactive prompt optimization during the "Awareness Loop".
- [x] Implement `bridging_grounding` mechanism to map symbolic symbols to latent vectors.

### Sprint 3: Active Inference (Orchestrator)
- [x] Implement `Expected Free Energy (G)` calculation in `orchestrator.py`.
- [x] Refactor `aprocess_turn` to use EFE for policy selection (Pragmatic vs. Epistemic value).
- [x] Add "Proactive Branching" trigger based on predicted ambiguity.
- [x] Implement `metacognition.py` "Dissonance Trigger" ($ |\text{Right\_Pred} - \text{Left\_Exp}| $).

### Sprint 4: Evolution Loop & Entropy Check
- [x] Refactor `evolution.py` to use Bayesian Optimization for prompt/parameter tuning.
- [x] Execute `sleep_mode` with full memory consolidation into the Knowledge Graph (adapters).
- [x] Run final entropy verification and harness check.

## Decision Log
- **2026-03-30**: Initial plan creation for V2 Evolution. Focus on Active Inference symmetry.
