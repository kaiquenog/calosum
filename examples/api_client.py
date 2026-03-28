"""
Example HTTP Client for the Calosum Cognitive Agent API.
Certifique-se de que o servidor FastAPI está operante em um terminal separado:
`python3 -m calosum.bootstrap.api`
"""

import urllib.request
import json
import uuid

def send_chat(text: str):
    url = "http://localhost:8000/v1/chat/completions"
    payload = json.dumps({
        "session_id": str(uuid.uuid4())[:8],
        "text": text
    }).encode("utf-8")
    
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    print(f"Sending request to {url}...\n> {text}\n")
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            print("Response:")
            # Display atomic symbolic actions outputted by the left hemisphere
            actions = result.get("result", {}).get("left_result", {}).get("actions", [])
            for a in actions:
                print(f"[{a.get('action_type')}] => {a.get('payload')}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    msg = sys.argv[1] if len(sys.argv) > 1 else "Estou muito preocupado, escreva um plano curto de emergência!"
    send_chat(msg)
