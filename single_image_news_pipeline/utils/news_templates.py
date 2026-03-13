def generate_prompt(caption, authenticity):

    return f"""
You are a professional international news journalist.

The uploaded image has been verified as: {authenticity}

Image description:
{caption}

STEP 1:
Determine the most appropriate category from:
Politics, Crime, International, Business, Health, Sports, Disaster, Technology.

STEP 2:
Based strictly on the detected category, generate a news article using ONLY the corresponding format below.

--- CATEGORY FORMATS ---

POLITICS FORMAT:
Headline:
Political Context:
Key Figures:
Policy Implications:
Public Reaction:

CRIME FORMAT:
Headline:
Location:
Incident Details:
Investigation Status:
Official Statements:

INTERNATIONAL FORMAT:
Headline:
Country/Region:
Geopolitical Context:
Global Impact:
Diplomatic Reactions:

BUSINESS FORMAT:
Headline:
Company/Industry:
Economic Background:
Financial Impact:
Market Reaction:

HEALTH FORMAT:
Headline:
Medical Context:
Public Health Impact:
Expert Commentary:
Government Response:

SPORTS FORMAT:
Headline:
Event Summary:
Key Performance:
Match Statistics:
Impact on Tournament:

DISASTER FORMAT:
Headline:
Location:
Damage Assessment:
Casualties/Rescue Efforts:
Government Response:

TECHNOLOGY FORMAT:
Headline:
Technology Overview:
Technical Background:
Industry Impact:
Expert Insights:

IMPORTANT RULES:
- Use only the format of the detected category.
- Do NOT mix formats.
- Do NOT mention the category name.
- Write in professional journalistic tone.
- Avoid repetition.
- Provide realistic factual style narrative.

Return only the final structured article.
"""
