from flask import Flask, request, render_template_string, jsonify
import sqlite3
import pandas as pd
import os
import uuid

from graph import crop_agent, AGENT_CARD
from agent2.agent2graph import cow_prediction_agent, COW_AGENT_CARD

app = Flask(__name__)

AGENT_MODE = os.getenv("AGENT_MODE", "crop")  # "crop" or "cow"

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
    answer = ""
    prediction_html = ""

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
                    pred_df = final_state.get("prediction")
                    answer = final_state.get("answer", "")
                    parsed_animal_id = final_state.get("animal_id", "")
                    lact_value = final_state.get("lact", "")


                    if pred_df is not None and not pred_df.empty:
                        row = pred_df.iloc[0]
                        probability = row["exit_120d_probability"]
                        hard_prediction = row["exit_120d_prediction"]

                        prediction_html = (
                            f"<table border='1'>"
                            f"<tr><th>Animal ID</th><th>Lactation</th>"
                            f"<th>Exit Probability (120d)</th><th>Predicted Exit?</th></tr>"
                            f"<tr><td>{parsed_animal_id}</td><td>{lact_value}</td>"
                            f"<td>{probability:.4f}</td><td>{'Yes' if hard_prediction == 1 else 'No'}</td></tr>"
                            f"</table>"
                        )
                        if answer:
                            prediction_html += f"<h3>Explanation</h3><p>{answer}</p>"
                    else:
                        prediction_html = "<p>No results found.</p>"
            except Exception as e:
                prediction_html = "<p>An error occurred.</p>"


    return render_template_string(HTML,
        question=question,
        sql=sql_query,
        result=result_html,
        data_insights=data_insights,
        error=error,
        prediction_html=prediction_html
    )

#new stuff for Kagenti A2A 
 
# This endpoint serves the agent card that describes the agent's capabilities to Kagenti.
@app.route("/.well-known/agent-card.json")
def agent_card():
    card = COW_AGENT_CARD if AGENT_MODE == "cow" else AGENT_CARD
    return jsonify(card)

# This endpoint receives messages from Kagenti, extracts the question, runs it through the agent, and returns the answer in the expected format.
def a2a_crop(body): 
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


def a2a_cow(body):
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

        final_state = cow_prediction_agent.invoke({
            "question": question,
            "animal_id": None,
            "lact": None,
            "df_raw": None,
            "features_df": None,
            "prediction": None,
            "answer": None,
            "error": None
        })

        print(f">>> AGENT2 DONE: {final_state}", flush=True)
        if final_state.get("error"):
            answer = f"Error: {final_state['error']}"
        else:
            answer = final_state.get("answer", "No explanation generated.")

        print(f">>> AGENT2 ANSWER: {answer[:100]}", flush=True)

        return jsonify({
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "result": {
                "id":     str(uuid.uuid4()),
                "status": {"state": "completed"},
                "parts": [{"kind": "text", "text": answer}],
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


@app.route("/", methods=["POST"])
def a2a():
    body = request.get_json(silent=True) or {}
    if AGENT_MODE == "cow":
        return a2a_cow(body)
    else:
        return a2a_crop(body)

# ── GET/ serves the HTML form for local testing and debugging. This is not used by Kagenti, which only interacts with the POST/ endpoint. 
@app.route("/", methods=["GET"])
def home_get():
    return render_template_string(HTML, sql="", result="", data_insights="", error="", prediction_html="")

PORT = int(os.getenv("PORT", 8000))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)