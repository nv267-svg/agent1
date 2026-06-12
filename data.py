import pandas as pd 
import sqlite3 
import os
import random

DB_PATH  = os.getenv("DB_PATH", "crop.db")
CSV_PATH = "crop_yield.csv"

if os.path.exists(CSV_PATH):
    # Local dev — use the real CSV
    df = pd.read_csv(CSV_PATH)
else:
    # In container — generate fake data
    print("CSV not found, generating fake data...")
    random.seed(42)
    regions = ["North","South","East","West","Central"]
    soils   = ["Clay","Sandy","Loam","Silt","Peat"]
    crops   = ["Wheat","Rice","Maize","Barley","Soybean","Cotton","Sugarcane","Potato"]
    weather = ["Sunny","Rainy","Cloudy","Windy","Humid"]

    df = pd.DataFrame([{
        "Region":                 random.choice(regions),
        "Soil_Type":              random.choice(soils),
        "Crop":                   random.choice(crops),
        "Rainfall_mm":            round(random.uniform(300, 1500), 1),
        "Temperature_Celsius":    round(random.uniform(15, 35), 1),
        "Fertilizer_Used":        random.choice(["Yes", "No"]),
        "Irrigation_Used":        random.choice(["Yes", "No"]),
        "Weather_Condition":      random.choice(weather),
        "Days_to_Harvest":        random.randint(60, 180),
        "Yield_tons_per_hectare": round(random.uniform(1, 8), 2),
    } for _ in range(1000)])

conn = sqlite3.connect(DB_PATH)
df.to_sql("crop", conn, if_exists="replace", index=False)
conn.close()
print(f"crop.db ready — {len(df)} rows.")