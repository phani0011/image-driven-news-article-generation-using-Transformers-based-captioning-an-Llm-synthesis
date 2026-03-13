import requests


class NewsGenerator:

    def __init__(self):
        self.api_key = "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"
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def generate_news(self, prompt):

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a professional journalist."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.6
        }

        response = requests.post(self.url, headers=headers, json=payload)

        if response.status_code != 200:
            return "Error generating article."

        return response.json()["choices"][0]["message"]["content"]
