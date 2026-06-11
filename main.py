import os
import uuid
import sqlite3
import pandas as pd
from flask import Flask, request, jsonify
from text_to_sql import generate_sql

app = Flask(__name__)

AGENT_NAME = "crop1"
AGENT_VERSION = "0.0.1"
AGENT_DESCRIPTION = (
    "A crop data agent that answers questions about crop yields, "
    "soil types, weather conditions, and more using natural language to SQL."
)
AGENT_ENDPOINT = os.getenv("AGENT_ENDPOINT", "http://localhost:8080")

AGENT_CARD = {
    "name": AGENT_NAME,
    "description": AGENT_DESCRIPTION,
    "url": AGENT_ENDPOINT,
    "version": AGENT_VERSION,
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": False,
    },
    "skills": [
        {
            "id": "crop-query",
            "name": "Crop Data Query",
            "description": "Answer questions about crop data using natural language.",
            "tags": ["crop", "agriculture", "sql", "data"],
            "examples": [
                "What is the average yield for wheat?",
                "Which crops grow best in sandy soil?",
                "Show crops with rainfall above 200mm",
            ],
        }
    ],
}


# Support both discovery URLs
@app.route("/.well-known/agent.json", methods=["GET"])
@app.route("/.well-known/agent-card.json", methods=["GET"])
def agent_card():
    return jsonify(AGENT_CARD), 200


def run_task(question: str) -> dict:
    try:
        sql_query = generate_sql(question)

        conn = sqlite3.connect("/app/crop.db")
        df = pd.read_sql_query(sql_query, conn)
        conn.close()

        if df.empty:
            answer_text = "No results found for that query."
        else:
            answer_text = df.to_json(orient="records")

        return {
            "status": "completed",
            "sql": sql_query,
            "answer": answer_text,
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
        }


@app.route("/", methods=["GET", "POST"])
def handle_root():

    if request.method == "GET":
        return jsonify({
            "status": "healthy",
            "agent": AGENT_NAME
        }), 200

    body = request.get_json(silent=True) or {}

    if "method" in body:

        method = body.get("method", "")
        params = body.get("params", {})
        task_id = params.get("id") or str(uuid.uuid4())

        if method not in ("tasks/send", "tasks/sendSubscribe", "message/send", "message/sendSubscribe"):
            return jsonify({
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                },
            }), 404

        message = params.get("message", {})
        parts = message.get("parts", [])

        question = ""
        for part in parts:
            if part.get("type") == "text":
                question = part.get("text", "")
                break

        if not question.strip():
            return jsonify({
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {
                    "code": -32602,
                    "message": "No question text found in message parts"
                },
            }), 400

        result = run_task(question)

        task_state = (
            "completed"
            if result["status"] == "completed"
            else "failed"
        )

        artifact_parts = []

        if result["status"] == "completed":
            artifact_parts = [
                {
                    "type": "text",
                    "text": result["answer"]
                },
                {
                    "type": "text",
                    "text": f"SQL: {result['sql']}"
                }
            ]

        return jsonify({
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "result": {
                "id": task_id,
                "status": {
                    "state": task_state
                },
                "artifacts": (
                    [{"parts": artifact_parts}]
                    if artifact_parts
                    else []
                ),
            },
        }), 200

    question = body.get("question", "")

    if not question.strip():
        return jsonify({
            "error": "no question provided"
        }), 400

    result = run_task(question)

    if result["status"] == "failed":
        return jsonify({
            "error": result["error"]
        }), 500

    return jsonify({
        "sql": result["sql"],
        "answer": result["answer"],
    }), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(
        host="0.0.0.0",
        port=port,
    )
