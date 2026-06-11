import os
import requests

schema = """
Table: crop

Columns:
Region
Soil_Type
Crop
Rainfall_mm
Temperature_Celsius
Fertilizer_Used
Irrigation_Used
Weather_Condition
Days_to_Harvest
Yield_tons_per_hectare
"""


def generate_sql(question):
    api_base = os.getenv("LLM_API_BASE", "http://localhost:11434/v1")
    model = os.getenv("LLM_MODEL", "llama3")

    prompt = f"""
    You are a SQLite SQL generator.

    Return ONLY the raw SQL query, no explanations, no markdown, no backticks.
    Return ONLY the columns asked for.

    Schema:
    {schema}

    Question:
    {question}
    """

    response = requests.post(
        f"{api_base}/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('LLM_API_KEY', 'dummy')}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()
