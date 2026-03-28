import os
import json
import httpx
from calosum.adapters.llm_payloads import left_hemisphere_result_schema

schema = left_hemisphere_result_schema()

url = "https://api.openai.com/v1/chat/completions"
headers = {"Authorization": f"Bearer {os.environ.get('CALOSUM_LEFT_API_KEY')}"}
payload = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "hi"}],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "test",
            "strict": True,
            "schema": schema
        }
    }
}

resp = httpx.post(url, headers=headers, json=payload)
print(resp.status_code, resp.text)
