from __future__ import annotations
from dataclasses import dataclass, field

@dataclass(slots=True)
class BranchingBudget:
    max_width: int = 3
    max_depth: int = 1

@dataclass(slots=True)
class CalosumAgentConfig:
    max_runtime_retries: int = 2
    surprise_threshold: float = 0.6
    awareness_interval_turns: int = 1
    episode_volume_threshold: int = 50 # V3 Trigger
    branching_budget: BranchingBudget = field(default_factory=BranchingBudget)
