import psycopg


conn = psycopg.connect(
    dbname="FotF",
    user="postgres",
    password="farmdata2024",
    host="localhost", 
    port="5433"
)

cur = conn.cursor()

cur.execute("SELECT * FROM aggregated_data.one_row_per_cow_per_day")

rows = cur.fetchall()

for row in rows:
    print(row)

cur.close()
conn.close()