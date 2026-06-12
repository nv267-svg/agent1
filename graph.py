from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import os
import sqlite3
import pandas as pd
import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.2:3b-instruct-fp16")
DB_PATH         = os.getenv("DB_PATH",         "crop.db")

SCHEMA = """
Table: crop

Columns:
Region
Soil_Type
Crop
Rainfall_mm
Temperature_Celsius
Fertilizer_Used
Irrigation_Used
Weather_Condition
Days_to_Harvest
Yield_tons_per_hectare
"""

class AgentState(TypedDict):
    question: str
    sql_query: Optional[str]
    rows: Optional[list[dict]]
    error: Optional[str]
    
#node1: generate SQL from question
def generate_sql_node(state: AgentState) -> AgentState:
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    prompt = f"""
    You are a SQLite SQL generator.

    Return ONLY SQL.
    Return ONLY the columns asked for.

    Schema:
    {SCHEMA}

    Question:
    {state['question']} 
    """
# ^^ state[question] accesses the question key in the state dictionary.

    try: 
        response = llm.invoke(([HumanMessage(content=prompt)]))
        sql = response.content.strip()
        sql = clean(sql)
        return {**state, "sql_query": sql, "error": None} #copies the dictionary and edits the sql_query and error keys
    except Exception as e:
        return {**state, "sql_query": None, "error": f"Error generating SQL: {str(e)}"}
    
#node2: execute SQL and return results
def execute_sql_node(state: AgentState) -> AgentState:
     if state.get("error") or not state.get("sql_query"):
        return state
     try:
         conn = sqlite3.connect(DB_PATH)
         conn.row_factory = sqlite3.Row #returns rows as dictionaries
         df = pd.read_sql_query(state["sql_query"], conn)         
         conn.close()
         rows = df.head(100).to_dict(orient="records") #converts dataframe to list of dictionaries
         return {**state, "rows": rows, "error": None} #copies the dictionary and edits the rows and error keys, and RETURNS
     except Exception as e:       
            return {**state, "rows": None, "error": f"SQL Error: {str(e)}"}


def build_graph():
     graph = StateGraph(AgentState)
     graph.add_node("generate_sql", generate_sql_node)
     graph.add_node("execute_sql", execute_sql_node)

     graph.add_edge(START, "generate_sql")
     graph.add_edge("generate_sql", "execute_sql")
     graph.add_edge("execute_sql", END)

     return graph.compile()


#compiles the graph and returns an agent that can be invoked with an initial state
crop_agent = build_graph() 




def clean(sql: str) -> str:
    if sql.startswith("```"):
            sql = sql.split("```")[1]
            if sql.lower().startswith("sql"):
                sql = sql[3:]
            sql = sql.strip()
    return sql