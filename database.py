import psycopg


conn = psycopg.connect(
    dbname="FotF",
    user="postgres",
    password="farmdata2024",
    host="localhost", 
    port="5433"
)

cur = conn.cursor()

cur.execute("SELECT * FROM aggregated_data.daily_cow")

rows = cur.fetchall()

for row in rows:
    print(row)

cur.close()
conn.close()