import os
import glob

# 1. File renames
file_moves = {
    "src/calosum/domain/cognition/right_hemisphere.py": "src/calosum/domain/cognition/input_perception.py",
    "src/calosum/domain/cognition/left_hemisphere.py": "src/calosum/domain/cognition/action_planner.py",
    "src/calosum/domain/execution/runtime.py": "src/calosum/domain/execution/tool_runtime.py",
    "src/calosum/adapters/execution/action_runtime.py": "src/calosum/adapters/execution/tool_runtime.py",
    "src/calosum/adapters/hemisphere/left_hemisphere_rlm.py": "src/calosum/adapters/hemisphere/action_planner_rlm.py",
    "src/calosum/adapters/hemisphere/right_hemisphere_heuristic_jepa.py": "src/calosum/adapters/hemisphere/input_perception_heuristic_jepa.py",
    "src/calosum/adapters/hemisphere/right_hemisphere_hf.py": "src/calosum/adapters/hemisphere/input_perception_hf.py",
    "src/calosum/adapters/hemisphere/right_hemisphere_jepars.py": "src/calosum/adapters/hemisphere/input_perception_jepars.py",
    "src/calosum/adapters/hemisphere/right_hemisphere_trained_jepa.py": "src/calosum/adapters/hemisphere/input_perception_trained_jepa.py",
    "src/calosum/adapters/hemisphere/right_hemisphere_vjepa21.py": "src/calosum/adapters/hemisphere/input_perception_vjepa21.py",
    "src/calosum/adapters/hemisphere/right_hemisphere_vljepa.py": "src/calosum/adapters/hemisphere/input_perception_vljepa.py",
}

for src, dst in file_moves.items():
    if os.path.exists(src):
        os.rename(src, dst)
        print(f"Moved {src} to {dst}")
    else:
        print(f"Warning: {src} not found")

# 2. String replacements
replacements = {
    "RightHemisphereState": "InputPerceptionState",
    "CognitiveBridgePacket": "PerceptionSummary",
    "LeftHemisphereResult": "ActionPlannerResult",
    "RightHemispherePort": "InputPerceptionPort",
    "LeftHemispherePort": "ActionPlannerPort",
    "ActionRuntimePort": "ToolRuntimePort",
    "StrictLambdaRuntime": "ToolRuntime",
    "StrictLambdaRuntimeConfig": "ToolRuntimeConfig",
    "RightHemisphereJEPA": "InputPerceptionJEPA",
    "RightHemisphereJEPAConfig": "InputPerceptionJEPAConfig",
    "LeftHemisphereLogicalSLM": "ActionPlannerLogicalSLM",
    "LeftHemisphereLogicalSLMConfig": "ActionPlannerLogicalSLMConfig",
    
    # Import path replacements
    "calosum.domain.cognition.right_hemisphere": "calosum.domain.cognition.input_perception",
    "calosum.domain.cognition.left_hemisphere": "calosum.domain.cognition.action_planner",
    "calosum.domain.execution.runtime": "calosum.domain.execution.tool_runtime",
    "calosum.adapters.execution.action_runtime": "calosum.adapters.execution.tool_runtime",
    "calosum.adapters.hemisphere.left_hemisphere_rlm": "calosum.adapters.hemisphere.action_planner_rlm",
    "calosum.adapters.hemisphere.right_hemisphere_heuristic_jepa": "calosum.adapters.hemisphere.input_perception_heuristic_jepa",
    "calosum.adapters.hemisphere.right_hemisphere_hf": "calosum.adapters.hemisphere.input_perception_hf",
    "calosum.adapters.hemisphere.right_hemisphere_jepars": "calosum.adapters.hemisphere.input_perception_jepars",
    "calosum.adapters.hemisphere.right_hemisphere_trained_jepa": "calosum.adapters.hemisphere.input_perception_trained_jepa",
    "calosum.adapters.hemisphere.right_hemisphere_vjepa21": "calosum.adapters.hemisphere.input_perception_vjepa21",
    "calosum.adapters.hemisphere.right_hemisphere_vljepa": "calosum.adapters.hemisphere.input_perception_vljepa",
}

python_files = glob.glob("src/**/*.py", recursive=True) + glob.glob("tests/**/*.py", recursive=True)

for filepath in python_files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        new_content = content
        for old, new in replacements.items():
            new_content = new_content.replace(old, new)
            
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated terms in {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

