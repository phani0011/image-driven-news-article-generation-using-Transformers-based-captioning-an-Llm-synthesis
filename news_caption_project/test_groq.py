import requests

api_key = "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"

url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "llama-3.1-8b-instant",   # UPDATED MODEL
    "messages": [
        {"role": "system", "content": "You are a journalist."},
        {"role": "user", "content": "Write a short technology news headline."}
    ],
    "temperature": 0.5,
    "max_tokens": 100
}

response = requests.post(url, headers=headers, json=payload)

print("Status Code:", response.status_code)
print("Response:\n", response.text)
