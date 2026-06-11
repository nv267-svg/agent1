from flask import Flask, request, render_template_string
import sqlite3
import pandas as pd

from graph import crop_agent

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Crop Yield Agent</title>
</head>
<body>

    <h1>Text-to-SQL for Crop Yield</h1>

    <form method="POST">
        <input
            type="text"
            name="question"
            placeholder="Ask a question..."
            style="width:500px;"
            required
        >
        <button type="submit">Submit</button>
    </form>

    {% if sql %}
        <h2>Generated SQL</h2>
        <pre>{{ sql }}</pre>
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

@app.route("/", methods=["GET", "POST"])
def home():

    sql_query = ""
    result_html = ""
    error = ""

    if request.method == "POST":

        question = request.form["question"]

        try:
            final_state = crop_agent.invoke({
                "question": question,
                "sql_query": None,
                "rows": None,
                "error": None
            })

            if final_state.get("error"):
                error = final_state["error"]
            else: 
                sql_query = final_state.get("sql_query", "")
                rows = final_state.get("rows") or []
                if rows:
                    df = pd.DataFrame(rows)
                    result_html = df.to_html(index=False)
                else:
                    result_html = "<p>No results found.</p>"
        except Exception as e:
            error = str(e)

    return render_template_string(
        HTML,
        sql=sql_query,
        result=result_html,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)