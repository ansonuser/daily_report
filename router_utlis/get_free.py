import requests
import os
from dotenv import load_dotenv
# import pdb


load_dotenv()


def score_model_for_summary(description: str) -> int:
    keywords = {
        "instruct": 2,
        "long-context": 2,
        "reasoning": 2,
        "math": 1,
        "uncensored": -3,
        "jailbreak": -3
    }
    
    desc = description.lower()
    score = 0
    
    for word, val in keywords.items():
        if word in desc:
            score += val 
    return score 


def get_free_models():
    url = "https://openrouter.ai/api/v1/models"

    headers = {
        "Authorization": f"Bearer {os.getenv('openrouter_key')}",
        "User-Agent": "Mozilla/5.0"
    }
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"❌ Failed to fetch models: {resp.status_code}")
        return []

    models = resp.json()
    free_models = []

    # pdb.set_trace()
    for m in models["data"]:
        # print(m)
        if m.get("pricing", {}).get("prompt") == '0' and m.get("pricing", {}).get("completion") == '0':
            score = score_model_for_summary(m.get("description", ""))
            if score > 2:
                free_models.append({
                    "id": m["id"],
                    "name": m.get("name", ""),
                    "description": m.get("description", ""),
                    "context_length": m.get("context_length", ""),
                    "score": score
                })

    free_models.sort(key=lambda x: x["context_length"], reverse=True)
    return free_models

if __name__ == "__main__":
    free = get_free_models()
    print(f"✅ 共找到 {len(free)} 個免費模型：\n")
    for m in free:
        print(f"- 🆓 {m['name']} ({m['id']})\n  {m['description']}\n")
