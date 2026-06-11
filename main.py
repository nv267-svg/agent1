import sys
import sqlite3
import pandas as pd
from text_to_sql import generate_sql
import requests
def handle_kagenti_request():
    if len(sys.argv) < 2:
        print("Error: No question provided by Kagenti interface.")
        sys.exit(1)
        
    user_question = " ".join(sys.argv[1:])
    
    try:
        sql_query = generate_sql(user_question)
        
        conn = sqlite3.connect("crop.db")
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        
        if df.empty:
            print("No matching data found in the crop database.")
        else:
            print(df.to_string(index=False))
            
    except Exception as e:
        print(f"An execution error occurred: {str(e)}")

if __name__ == "__main__":
    handle_kagenti_request()