import requests

payload = {
    "model": "mistral",
    "prompt": "请总结以下情绪：happy: 5, sad: 2, angry: 1",
    "stream": False
}

try:
    r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
    r.raise_for_status()
    print("✅ 响应内容：", r.json()["response"])
except Exception as e:
    print("❌ 请求失败：", str(e))