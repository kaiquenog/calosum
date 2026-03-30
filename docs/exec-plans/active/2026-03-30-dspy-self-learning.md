# DSPy Self-Learning via Night Trainer (Sleep Mode)

## Purpose
Evolve the Calosum architecture by fully integrating the DSPy framework into the "Sleep Mode" (Night Trainer). This will enable the agent to automatically extract successful episodes from its episodic memory (`episodic.jsonl`) and use them to optimize the system prompts (via `MIPROv2` or `BootstrapFewShot`) for the Left Hemisphere. This transitions the agent from static, hard-coded prompts to a self-improving pipeline.

## Scope
- Update `src/calosum/adapters/night_trainer_dspy.py` to fetch valid episodic memories as a training set.
- Implement a DSPy optimizer (e.g., `MIPROv2`) that uses the agent's internal metacognitive metrics (like safety, empathy priority from the GEA Reflection Controller) or simply the `tool_success_rate` as a reward metric.
- Save the compiled DSPy program (the optimized prompt/few-shot examples) back to disk so `llm_payloads.py` and the `QwenLeftHemisphereAdapter` can load it on the next startup.

## Validation
- Ensure no regressions occur by running the standard unit tests (`PYTHONPATH=src python3 -m unittest discover -s tests -t .`).
- Run `PYTHONPATH=src python3 -m calosum.harness_checks` to guarantee the changes respect the *Ports and Adapters* architectural boundaries defined in `MODULE_RULES`.
- Simulate a Sleep Mode run via CLI (`python3 -m calosum.bootstrap.cli sleep`) to verify that the compiled artifact is generated and saved correctly.

## Progress
- [ ] Create the initial execution plan (this document).
- [ ] Implement the `Memory-to-DSPy-Dataset` exporter/loader.
- [ ] Write the DSPy metric function based on metacognitive signals.
- [ ] Implement the optimization loop using `BootstrapFewShot` or `MIPROv2`.
- [ ] Wire the saved output to be loaded by the `LeftHemisphere`.
- [ ] Run tests and harness checks.

## Decision Log
- To be filled during implementation. Focus will be on prompt optimization rather than fine-tuning weights to avoid catastrophic forgetting and maintain architectural simplicity.
