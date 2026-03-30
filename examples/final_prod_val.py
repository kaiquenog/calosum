import requests
import json
import time

API_URL = "http://localhost:8000"

def test_v3_multimodal():
    print("\n--- Testing V3 Multimodal Perception (API) ---")
    payload = {
        "session_id": "prod-val-v3",
        "text": "Analise o que voce esta vendo no fluxo de video.",
        "signals": [
            {"modality": "video", "source": "camera_api", "payload": "mock_base64_data"}
        ]
    }
    
    response = requests.post(f"{API_URL}/v1/chat/completions", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json().get("result", {})
        right_state = result.get("right_state", {})
        surprise = right_state.get("surprise_score")
        richness = right_state.get("world_hypotheses", {}).get("visual_richness")
        print(f"Active Inference - Surprise: {surprise}, Visual Richness: {richness}")
    else:
        print(f"Error: {response.text}")
        
    return response.status_code == 200

def test_v3_mente_tab():
    print("\n--- Testing 'Mente' Tab Data (Telemetry) ---")
    response = requests.get(f"{API_URL}/v1/telemetry/dashboard/prod-val-v3")
    if response.status_code == 200:
        events = response.json().get("dashboard", [])
        felt_events = [e for e in events if e.get("type") == "felt"]
        print(f"Found {len(felt_events)} 'felt' events.")
        if felt_events:
            last_event = felt_events[-1]
            data = last_event.get("data", {})
            print(f"V3 Metrics - EFE: {data.get('expected_free_energy')}, Surprise: {data.get('surprise_score')}")
    return response.status_code == 200

if __name__ == "__main__":
    if test_v3_multimodal():
        test_v3_mente_tab()
