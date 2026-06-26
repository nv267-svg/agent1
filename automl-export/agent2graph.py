import pandas as pd
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from sqlalchemy import create_engine
import os
import pandas as pd
from build_features import build_cow_lactation_features
from inference_example import predict


# Agent pulls this from Postgres -- raw, unaggregated daily rows

try:
    db_url = "postgresql+psycopg://postgres:farmdata2024@128.84.40.194:5432/FotF"
    engine = create_engine(db_url)      

    df_raw = pd.read_sql("""
        SELECT * FROM aggregated_data.one_row_per_cow_per_day
        WHERE animal_id = '2075' AND lact = '6'
        AND dim BETWEEN 1 AND 21
    """, engine, params={"animal_id": 2075, "lact": 6})

except Exception as e:
    print(f"Error connecting to database: {e}")
    df_raw = pd.DataFrame()  # Empty DataFrame if connection fails

features_df = build_cow_lactation_features(df_raw, min_early_records=0)
# min_early_records=0 so a single sparse cow doesn't get silently dropped

print(df_raw['daily_weight'].dtype, df_raw['daily_weight'].isna().all())
print(df_raw['yesterday_s_weight'].dtype, df_raw['yesterday_s_weight'].isna().all())

print(f"\nBuilt feature row:")
print(features_df.T)  # transposed so you can read all ~116 features vertically

result = predict(features_df)
print(f"\nPrediction:")
print(result)