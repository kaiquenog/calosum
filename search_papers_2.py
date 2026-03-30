import sys
import json
import urllib.request
import time

def search(query):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query.replace(' ', '+')}&limit=5&fields=title,url,abstract,year"
    try:
        time.sleep(3) # Respect rate limits
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode('utf-8'))
        for paper in data.get('data', []):
            title = paper.get('title')
            year = paper.get('year')
            url = paper.get('url')
            abstract = str(paper.get('abstract') or '')[:300]
            print(f"- **{title}** ({year})\n  URL: {url}\n  Abstract: {abstract}...\n")
    except Exception as e:
        print(f"Error: {e}")

print("--- Active Inference + LLM ---")
search("active inference LLM autonomous agent")
print("--- Dual Process (System 1 / System 2) ---")
search("dual process system 1 system 2 language model")
