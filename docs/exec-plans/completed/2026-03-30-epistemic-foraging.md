# Epistemic Foraging via Active Inference

## Purpose
Evolve the agent from a reactive LLM to an autonomous epistemic forager. When the Right Hemisphere detects high surprise or ambiguity (Free Energy), the Cognitive Tokenizer should translate this into an explicit directive for the Left Hemisphere to seek information (e.g., using `search_web`, `read_file` or `execute_bash`) before formulating a final response.

## Scope
- Update `src/calosum/domain/bridge.py` to inject epistemic foraging directives when `surprise_score` is high.
- This closes the loop of Active Inference, moving from passive surprise measurement to active uncertainty reduction.

## Validation
- Run `harness_checks.py` to ensure compliance.
- Run unit tests `PYTHONPATH=src python3 -m unittest discover -s tests -t .` to verify that no regressions occur.

## Progress
- [ ] Implement directive injection in `bridge.py`.
- [ ] Run tests and verify the harness checks.

## Decision Log
- Decided to use `surprise_score` >= 0.3 as a threshold for injecting the "epistemic foraging" directive, prompting the agent to actively reduce uncertainty before responding.
