# Endogenous Goal Generation (Telogenesis)

## Purpose
Align the agent with SOTA Active Inference models by allowing it to generate its own goals. Instead of only reacting to user prompts, the Orchestrator should feature an `idle_foraging` (Background Foraging Mode) capability. When called (e.g., during idle periods or as a periodic task), the agent inspects its `KnowledgeGraph` for staleness or ambiguity and generates a "self-prompt" to actively research and update its memory (epistemic foraging).

## Scope
- Update `src/calosum/domain/orchestrator.py` (`CalosumAgent`) to include an `idle_foraging` or `abackground_forage` method.
- This method will request a snapshot of the knowledge graph, identify areas with low density/staleness, create a synthetic `UserTurn` (self-prompt), and route it through the normal execution engine (`run_variant`) with a specific background variant label.
- The outcome of this execution will automatically update the episodic memory and subsequently semantic memory, effectively letting the agent learn without human interaction.

## Validation
- Ensure unit tests pass without breaking existing APIs (`PYTHONPATH=src python3 -m unittest discover -s tests -t .`).
- Ensure harness checks pass (`PYTHONPATH=src python3 -m calosum.harness_checks`).

## Progress
- [ ] Create this execution plan.
- [ ] Implement the `idle_foraging` logic in the Orchestrator.
- [ ] Connect CLI or tests to trigger it.
- [ ] Verify tests and harness checks.

## Decision Log
- To be filled during implementation. I will implement `aprocess_idle_turn` in `CalosumAgent` to reuse the existing `AgentExecutionEngine` safely.
