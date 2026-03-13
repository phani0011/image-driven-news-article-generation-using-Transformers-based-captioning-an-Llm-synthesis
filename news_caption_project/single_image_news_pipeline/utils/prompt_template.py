def create_news_prompt(caption, authenticity):

    prompt = f"""
You are an international news journalist.

Image authenticity: {authenticity}

Image description:
{caption}

Write a professional news article in formal tone.
"""

    return prompt
