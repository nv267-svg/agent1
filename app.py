from flask import Flask, request, render_template_string, jsonify
import sqlite3
import pandas as pd
import os
import uuid

from graph import crop_agent
from agent2.agent2graph import cow_prediction_agent

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>FotF Cow Data Agent</title>
</head>
<body>

    <h1>Text-to-SQL for FotF Cow Data </h1>

    <form method="POST" action="/ui">
    <input type="hidden" name="form_type" value="question">
    <input
        type="text"
        name="question"
        placeholder="Ask a question..."
        style="width:500px;"
        required
    >
    <button type="submit">Submit</button>
</form>

<h3>Cow Exit Predictor</h3>

<form method="POST" action="/ui">
    <input type="hidden" name="form_type" value="prediction">
    <input
        type="text"
        name="animal_id"
        placeholder="Enter cow animal_id..."
        style="width:500px;"
        required
    >
    <button type="submit">Submit</button>
</form>

    {% if question %}
        <h2>Question</h2>
        <pre>{{ question }}</pre>
    {% endif %}

    {% if sql %}
        <h2>Generated SQL</h2>
        <pre>{{ sql }}</pre>
    {% endif %}

    {% if result %}
        <h2>Results</h2>
        {{ result|safe }}
    {% endif %}

    {% if prediction_html %}
        <h2>Prediction</h2>
        {{ prediction_html|safe }}
    {% endif %}


</body>
</html>
"""

@app.route("/ping")
def ping():
    return "Flask is working", 200

@app.route("/ui", methods=["GET", "POST"])
def ui():

    question = ""
    sql_query = ""
    result_html = ""
    error = ""
    data_insights = ""
    prediction = ""
    answer = ""

    if request.method == "POST":

        form_type = request.form["form_type"]

        if form_type == "question":
            question = request.form["question"]
            print(f">>> Got question: {question}", flush=True)

            try:
                #the initial state of the agent is a dictionary with only the question so far
                print(">>> Invoking agent...", flush=True)
                final_state = crop_agent.invoke({ 
                    "question": question,
                    "sql_query": None,
                    "rows": None,
                    "data_insights": None,
                    "error": None
                })
                print(f">>> Agent finished: {final_state}", flush=True)

                if final_state.get("error"):
                    error = final_state["error"]
                else: 
                    sql_query = final_state.get("sql_query", "")
                    rows = final_state.get("rows") or []
                    data_insights = final_state.get("data_insights", "")
                    if rows:
                        df = pd.DataFrame(rows)
                        result_html = df.to_html(index=False)
                        result_html += f"<h3>Data Insights</h3><p>{data_insights}</p>"
                    else:
                        result_html = "<p>No results found.</p>"
            except Exception as e:
                error = str(e)

        elif form_type == "prediction":
            animal_id = request.form["animal_id"]
            print(f">>> Got animal_id: {animal_id}", flush=True)
            try:
                #the initial state of the agent is a dictionary with only the question so far
                print(">>> Invoking agent...", flush=True)
                final_state = cow_prediction_agent.invoke({
                    "question": animal_id,
                    "lact": None,
                    "df_raw": None,
                    "features_df": None,
                    "prediction": None,
                    "answer": None,
                    "error": None
                })
                print(f">>> Agent finished: {final_state}", flush=True)

                if final_state.get("error"):
                    error = final_state["error"]
                else: 
                    prediction = final_state.get("prediction", "")
                    answer = final_state.get("answer", "")
                    if prediction and answer:
                        df = pd.DataFrame([{"Prediction": prediction, "Answer": answer}])
                        prediction_insights = f"Prediction for animal_id {animal_id}: {prediction}, Actual Answer: {answer}"
                        prediction_html = df.to_html(index=False)
                        prediction_html += f"<h3>Prediction Insights</h3><p>{prediction_insights}</p>"
                    else:
                        prediction_html = "<p>No results found.</p>"
            except Exception as e:
                error = str(e)


    return render_template_string(HTML,
        question=question,
        sql=sql_query,
        result=result_html,
        data_insights=data_insights,
        error=error,
        prediction_html=prediction_html
    )

#new stuff for Kagenti A2A 


AGENT_CARD = {
    "name": "crop-yield-agent",
    "description": (
        "Text-to-SQL agent for crop yield data. "
        "Ask natural language questions about crop yields, regions, rainfall, and more."
    ),
    "version": "1.0.0",
    "url": "http://crop-yield-agent.team1.svc.cluster.local:8080",
    "capabilities": {"streaming": False},
    "defaultInputModes":  ["text"],
    "defaultOutputModes": ["text"],
    "skills": [
        {
            "id":          "crop_yield_query",
            "name":        "Crop Yield Query",
            "description": "Answer questions about crop yields using SQL over a SQLite database.",
            "tags":        ["sql", "agriculture", "crop", "yield"],
            "examples": [
                "What are the top 5 crops by average yield?",
                "Which region has the highest rainfall?",
                "Show me all crops grown with irrigation in the North region.",
            ],
        }
    ],
}
 
# This endpoint serves the agent card that describes the agent's capabilities to Kagenti.
@app.route("/.well-known/agent-card.json")
def agent_card():
    return jsonify(AGENT_CARD)

# This endpoint receives messages from Kagenti, extracts the question, runs it through the agent, and returns the answer in the expected format.
@app.route("/", methods=["POST"])
def a2a(): 
    body = request.get_json(silent=True) or {}
    print(f">>> FULL BODY: {body}", flush=True)
    
    try:
        if body.get("jsonrpc") != "2.0" or body.get("method") != "message/send":
            return jsonify({
                "jsonrpc": "2.0",
                "id":    body.get("id"),
                "error": {"code": -32601, "message": "Method not found"},
            }), 400

        parts    = body.get("params", {}).get("message", {}).get("parts", [])
        question = next((p["text"] for p in parts if p.get("type") == "text" or p.get("kind") == "text"), "")
        print(f">>> QUESTION: {question}", flush=True)

        if not question:
            return jsonify({
                "jsonrpc": "2.0",
                "id":    body.get("id"),
                "error": {"code": -32602, "message": "No text part found"},
            }), 400

        final_state = crop_agent.invoke({
            "question": question,
            "sql_query": None,   # was "sql"
            "rows":      None,
            "data_insights": None,
            "error":     None,
        })

        print(f">>> AGENT DONE: {final_state}", flush=True)

        if final_state.get("error"):
            answer = f"Error: {final_state['error']}"
        else:
            sql  = final_state.get("sql_query", "")
            rows = final_state.get("rows") or []
            data_insights = final_state.get("data_insights", "")
            answer = f"SQL:\n{sql}\n\nResults ({len(rows)} rows):\n{pd.DataFrame(rows).to_string(index=False)} + Data Insights: {data_insights}" if rows else f"SQL:\n{sql}\n\nNo rows returned."

        print(f">>> ANSWER: {answer[:100]}", flush=True)

        return jsonify({
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "result": {
                "id":     str(uuid.uuid4()),
                "status": {"state": "completed"},
                "parts": [{"kind": "text", "text": answer}],  # ← parts directly in result
            },
        })

    except Exception as e:
        print(f">>> EXCEPTION: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "error": {"code": -32603, "message": str(e)},
        }), 500


# ── GET/ serves the HTML form for local testing and debugging. This is not used by Kagenti, which only interacts with the POST/ endpoint. 
@app.route("/", methods=["GET"])
def home_get():
    return render_template_string(HTML, sql="", result="", data_insights="", error="")

PORT = int(os.getenv("PORT", 8000))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)