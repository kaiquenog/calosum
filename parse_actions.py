import sys, json
try:
    data = json.load(sys.stdin)
    actions = data.get("result", {}).get("left_result", {}).get("actions", [])
    for a in actions:
        print(f"Action Executed: {a.get('action_type')}")
except Exception as e:
    print(f"Error parsing JSON: {e}")
