import sqlite3
import pandas as pd
from flask import Flask, request, jsonify
from text_to_sql import generate_sql

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    question = request.get_json().get("question", "")
    
    if not question:
        return jsonify({"error": "no question provided"}), 400

    sql = generate_sql(question)
    
    conn = sqlite3.connect("crop.db")
    df = pd.read_sql_query(sql, conn)
    conn.close()
    
    return jsonify({
        "sql": sql,
        "answer": df.to_json(orient="records")
    })

if __name__ == "__main__":
    app.run(port=8080)
