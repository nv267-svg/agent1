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

    prompt = f"""
    You are a SQLite SQL generator.

    Return ONLY SQL.
    Return ONLY the columns asked for.

    Schema:
    {schema}

    Question:
    {question}
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]