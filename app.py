from flask import Flask, request, render_template_string
import sqlite3
import pandas as pd

from text_to_sql import generate_sql

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

@app.route("/", methods=["GET", "POST"])
def home():

    sql_query = ""
    result_html = ""
    error = ""

    if request.method == "POST":

        question = request.form["question"]

        try:
            sql_query = generate_sql(question)
            conn = sqlite3.connect("crop.db")
            df = pd.read_sql_query(sql_query, conn)
            result_html = df.head(100).to_html(index=False)

            conn.close()

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