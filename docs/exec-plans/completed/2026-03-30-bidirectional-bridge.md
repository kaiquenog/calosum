# Bidirectional Cognitive Bridge (System 2 overrides System 1)

## Purpose
In a true Neuro-Symbolic architecture (Dual Process Theory), System 2 (logical reasoning) must be able to act as a corrective regularizer for System 1 (neural perception/intuition). Currently, Calosum's Right Hemisphere dictates the `BridgeControlSignal` unconditionally via the `CognitiveTokenizer`. This plan implements a feedback loop where the logical execution engine can flag a "cognitive mismatch" (e.g., when the Right Hemisphere flagged "urgency" but the Left Hemisphere detected a casual interaction and successfully processed it without urgency). This feedback will be recorded as a reflection event to tune the neural bridge's weights.

## Scope
- Update `src/calosum/domain/orchestrator.py` or `src/calosum/domain/agent_execution.py` to detect when the Left Hemisphere successfully resolves a turn but explicitly notes a mismatch with the initial emotional/salience state provided by the Bridge.
- Send this feedback back to the `CognitiveTokenizer` via `record_reflection_event` or by directly updating the semantic memory with a corrective rule.
- This closes the bidirectional loop: System 1 primes System 2, and System 2 corrects System 1.

## Validation
- Run unit tests (`PYTHONPATH=src python3 -m unittest discover -s tests -t .`).
- Run architectural checks (`PYTHONPATH=src python3 -m calosum.harness_checks`).

## Progress
- [ ] Create this execution plan.
- [ ] Implement mismatch detection in the orchestrator/execution engine.
- [ ] Wire the feedback to the tokenizer/memory.
- [ ] Verify tests and harness checks.

## Decision Log
- To be filled during implementation. Focus will be on utilizing existing telemetry or reflection recording mechanisms in the `CognitiveTokenizer` to persist the correction without adding heavy new dependencies.
