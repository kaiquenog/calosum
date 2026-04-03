import os

filepath = "src/calosum/harness_checks.py"
replacements = {
    "domain.cognition.right_hemisphere": "domain.cognition.input_perception",
    "domain.cognition.left_hemisphere": "domain.cognition.action_planner",
    "domain.execution.runtime": "domain.execution.tool_runtime",
    "adapters.execution.action_runtime": "adapters.execution.tool_runtime",
    "adapters.hemisphere.left_hemisphere_rlm": "adapters.hemisphere.action_planner_rlm",
    "adapters.hemisphere.right_hemisphere_heuristic_jepa": "adapters.hemisphere.input_perception_heuristic_jepa",
    "adapters.hemisphere.right_hemisphere_hf": "adapters.hemisphere.input_perception_hf",
    "adapters.hemisphere.right_hemisphere_jepars": "adapters.hemisphere.input_perception_jepars",
    "adapters.hemisphere.right_hemisphere_trained_jepa": "adapters.hemisphere.input_perception_trained_jepa",
    "adapters.hemisphere.right_hemisphere_vjepa21": "adapters.hemisphere.input_perception_vjepa21",
    "adapters.hemisphere.right_hemisphere_vljepa": "adapters.hemisphere.input_perception_vljepa",
}

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()
    
for old, new in replacements.items():
    content = content.replace(old, new)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Updated terms in {filepath}")
