from flask import Flask, request, render_template_string, jsonify
import sqlite3
import pandas as pd
import os
import uuid

from graph import crop_agent, AGENT_CARD
from agent2.agent2graph import cow_prediction_agent, COW_AGENT_CARD
from orchestrator.orchestrator_graph import orchestrator_agent, ORCHESTRATOR_AGENT_CARD

app = Flask(__name__)

AGENT_MODE = os.getenv("AGENT_MODE", "crop")  

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

    {% if question %}
        <h2>Question</h2>
        <pre>{{ question }}</pre>
    {% endif %}

    {% if result %}
        <h2>Results</h2>
        {{ result|safe }}
    {% endif %}


</body>
</html>
"""

@app.route("/ping")
def ping():
    return "Flask is working", 200
@app.route("/ui", methods=["GET", "POST"])
def ui():
    final_output = ""
    if request.method == "POST":
        query = request.form.get("query")
        result = orchestrator_agent.invoke({"question": query})
        final_output = result.get("answer", "No response generated.")
        
    return render_template_string(HTML, final_output=final_output)

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
    ''''
    body = request.get_json(silent=True) or {}
    if AGENT_MODE == "cow":
       return a2a_cow(body)
    else:
       return a2a_crop(body)

    '''
    body = request.get_json(silent=True) or {}
    parts = body.get("params", {}).get("message", {}).get("parts", [])
    query = next((p["text"] for p in parts if p.get("type") in ["text", "kind"]), "")
    
    result = orchestrator_agent.invoke({"question": query})
    answer = result.get("answer", "No response generated.")
    
    return jsonify({
        "jsonrpc": "2.0",
        "id": body.get("id"),
        "result": {
            "status": {"state": "completed"},
            "parts": [{"kind": "text", "text": answer}]
        }
    })

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=True)