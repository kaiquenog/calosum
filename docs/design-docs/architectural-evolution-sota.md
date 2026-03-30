# Architectural Evolution: From LLM Application to Autonomous Agentic AI-OS

This document synthesizes the strategic architectural evolution plan for the Calosum project, heavily informed by recent SOTA research on Neuro-Symbolic Architectures, Dual Process Theory (System 1/System 2) in Large Language Models, Active Inference, and Metacognition (Group Evolving Agents/DSPy).

## 1. Context and The "Agentic Gap"

While Calosum possesses a strong foundation (Dual Hemisphere architecture, active inference scoring, and episodic memory), it currently borders on being a highly complex, reactive "LLM tool user" rather than a truly autonomous agent.

The SOTA literature highlights that true agentic behavior requires:
1.  **Epistemic Autonomy:** Agents should not just react to surprise; they should actively seek to reduce it before taking final action (Active Inference & Free Energy). *[Partially addressed via the recent Epistemic Foraging update]*
2.  **Bidirectional Dual Process:** "System 1" (Intuition/Embeddings) should guide "System 2" (Deliberation/LLM), but System 2 must be able to analytically override or tune System 1's heuristics when a logical error is detected (Neuro-Symbolic Integration).
3.  **Self-Evolving Metacognition:** The agent must rewrite its own operational parameters and prompts based on empirical success rates, removing human-authored "prompt engineering" (DSPy & Group Evolving Agents).

## 2. Near-Term Evolution Plan (Next Sprints)

### 2.1. Complete the "Self-Learning" Loop (DSPy Integration)
**Status:** Planned (Active: `docs/exec-plans/active/2026-03-30-dspy-self-learning.md`)
**Why:** Research shows that manual prompt engineering hits a ceiling. SOTA frameworks like DSPy treat LLM pipelines as compilable programs.
**Actionable Steps:**
- Activate `NightTrainerDSPyAdapter` in the Sleep Mode loop.
- Use `memory_qdrant` / `episodic.jsonl` to filter successful past interactions (`tool_success_rate == 1.0` and high user satisfaction).
- Run `dspy.MIPROv2` or `BootstrapFewShot` to mathematically optimize the `QwenLeftHemisphereAdapter` system prompts.
- Persist the compiled artifacts (JSON) so the agent "wakes up" smarter.

### 2.2. Bidirectional Cognitive Bridge (System 2 overriding System 1)
**Status:** Proposed Future Plan
**Why:** Current neuro-symbolic research emphasizes that System 2 (logical reasoning) must act as a corrective regularizer for System 1 (neural perception). Currently, Calosum's `RightHemisphere` dictates the `BridgeControlSignal` unconditionally.
**Actionable Steps:**
- Introduce a feedback loop where the `AgentExecutionEngine` (System 2 execution) can flag a "cognitive mismatch" (e.g., the Right Hemisphere flagged "urgency" but the Left Hemisphere detected a casual joke).
- Persist this mismatch as a new `SemanticRule` or feed it into the `CognitiveTokenizer`'s neural weights adaptation. This ensures the intuitive embedding model is "fine-tuned" by the logical outcomes.

### 2.3. Endogenous Goal Generation (Telogenesis)
**Status:** Proposed Future Plan
**Why:** Recent papers on Active Inference (e.g., *"Telogenesis: Goal Is All U Need"*) argue that agents should generate their own goals based on "epistemic gaps" (ignorance, surprise, staleness).
**Actionable Steps:**
- Upgrade the Orchestrator to support an "Idle Loop" or "Background Foraging Mode".
- When no user input is present, the agent reviews its `knowledge_graph.jsonl`. If it finds high uncertainty (staleness) in an important topic, it autonomously triggers a "self-prompt" to research that topic (web search) and updates its semantic rules.

## 3. Long-Term Aspirational Goal: The AI-OS

The ultimate goal, aligned with the original `INIT_PROJECT.MD` and the "AI-OS Self-Awareness" phase (Fase 5), is to shift Calosum from a "chatbot that uses tools" to a "Cognitive Operating System".

- **The Shift:** The human doesn't just "talk" to the agent; the human manages the agent's capacity, architecture, and goals via the UI (Architecture/Awareness dashboards).
- **The Engine:** The `GEAReflectionController` (Group Evolving Agents) becomes the core scheduler of the OS. It decides whether an incoming task requires a fast, cheap model (System 1 bypass) or a deep, multi-agent branch exploration (System 2 deliberation).

## 4. Alignment with Harness Engineering

To avoid entropy while implementing these SOTA concepts:
- **No sprawling modules:** All active inference enhancements will remain confined to `domain/right_hemisphere.py` and `domain/bridge.py`.
- **Mechanical Checks:** As the DSPy integration generates compiled prompts, the `harness_checks.py` will be updated to validate the existence and valid JSON structure of these artifacts.
- **Incremental Plans:** Each step (2.1, 2.2, 2.3) will have a dedicated, versioned `active/` plan document before any code is written.
