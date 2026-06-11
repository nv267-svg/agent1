import os
import sqlite3
import pandas as pd
from flask import Flask, request, jsonify
from text_to_sql import generate_sql

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def handle_root():
    if request.method == "GET":
        return jsonify({"status": "healthy"}), 200
    
    data = request.get_json()
    user_question = data.get("question", "") if data else ""
    
    if not user_question.strip():
        return jsonify({"error": "no question"}), 400
        
    try:
        sql_query = generate_sql(user_question)
        conn = sqlite3.connect("crop.db")
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        
        if df.empty:
            return jsonify({"answer": "not possible"}), 200
        
        return jsonify({
            "query": sql_query,
            "answer": df.to_dict(orient="records")
        }), 200
            
    except Exception as e:
        return jsonify({"error": f"An execution error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
