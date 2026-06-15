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
            "description": "Answer questions about crop data by converting text into SQL.",
            "tags": ["crop", "agriculture", "sql", "data"],
            "examples": [
                "What is the average yield for wheat?",
                "Which crops grow best in sandy soil?",
                "Show crops with rainfall above 200mm",
            ],
        }
    ],
}


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

        answer_text = (
            "No results found for that query."
            if df.empty
            else df.to_json(orient="records")
        )

        return {"status": "completed", "sql": sql_query, "answer": answer_text}

    except Exception as e:
        return {"status": "failed", "error": str(e)}


def extract_question(params: dict) -> str:
    parts = params.get("message", {}).get("parts", [])
    for part in parts:
        if part.get("kind") == "text" or part.get("type") == "text":
            if part.get("text", "").strip():
                return part["text"].strip()

    parts = params.get("parts", [])
    for part in parts:
        if part.get("kind") == "text" or part.get("type") == "text":
            if part.get("text", "").strip():
                return part["text"].strip()

    parts = params.get("input", {}).get("parts", []) if isinstance(params.get("input"), dict) else []
    for part in parts:
        if part.get("kind") == "text" or part.get("type") == "text":
            if part.get("text", "").strip():
                return part["text"].strip()

    for key in ("text", "input", "query", "question", "content"):
        val = params.get(key, "")
        if isinstance(val, str) and val.strip():
            return val.strip()

    return ""


@app.route("/", methods=["GET", "POST"])
def handle_root():

    if request.method == "GET":
        return jsonify({"status": "healthy", "agent": AGENT_NAME}), 200

    body = request.get_json(silent=True) or {}

    if "method" in body:
        method = body.get("method", "")
        params = body.get("params", {})
        task_id = params.get("id") or str(uuid.uuid4())
        rpc_id = body.get("id")

        if method not in (
            "tasks/send",
            "tasks/sendSubscribe",
            "message/send",
            "message/sendSubscribe",
        ):
            return jsonify({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }), 200

        question = extract_question(params)

        if not question:
            return jsonify({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": -32602, "message": "No question text found in message parts"},
            }), 200

        result = run_task(question)

        artifact_parts = []
        if result["status"] == "completed":
            artifact_parts = [
                {"type": "text", "text": result["answer"]},
                {"type": "text", "text": f"SQL: {result['sql']}"},
            ]

        return jsonify({
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "id": task_id,
                "status": {"state": result["status"]},
                "artifacts": [{"parts": artifact_parts}] if artifact_parts else [],
            },
        }), 200

    question = body.get("question", "")

    if not question.strip():
        return jsonify({"error": "no question provided"}), 400

    result = run_task(question)

    if result["status"] == "failed":
        return jsonify({"error": result["error"]}), 500

    return jsonify({"sql": result["sql"], "answer": result["answer"]}), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

