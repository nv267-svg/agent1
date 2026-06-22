from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from sqlalchemy import create_engine
import os
import sqlite3
import pandas as pd
import requests
import psycopg

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.2:3b-instruct-fp16")
DB_PATH         = os.getenv("DB_PATH",         "crop.db")

SCHEMA = """
Table: aggregated_data.one_row_per_cow_per_day

Column names: 
one_row_per_cow_per_day.date
one_row_per_cow_per_day.animal_id
one_row_per_cow_per_day.lact
one_row_per_cow_per_day.dim
one_row_per_cow_per_day.pen
one_row_per_cow_per_day.cbrd
one_row_per_cow_per_day.pta
one_row_per_cow_per_day.gdpr
one_row_per_cow_per_day.fdat
one_row_per_cow_per_day.cdat
one_row_per_cow_per_day.ddat
one_row_per_cow_per_day.rc
one_row_per_cow_per_day.rpro
one_row_per_cow_per_day.dcc
one_row_per_cow_per_day.scc
one_row_per_cow_per_day.ls
one_row_per_cow_per_day.ddry
one_row_per_cow_per_day.t2000
one_row_per_cow_per_day.psirc
one_row_per_cow_per_day.tbrd
one_row_per_cow_per_day.dc_events
one_row_per_cow_per_day.milkings_yesterday
one_row_per_cow_per_day.milk_yield_yesterday
one_row_per_cow_per_day.yield_yesterday_session_1
one_row_per_cow_per_day.yield_yesterday_session_2
one_row_per_cow_per_day.yield_yesterday_session_3
one_row_per_cow_per_day.avg_daily_yield_last_7d
one_row_per_cow_per_day.yesterday_deviation_pct_from_avg_last_7d
one_row_per_cow_per_day.total_yield_in_lactation
one_row_per_cow_per_day.milk_yield_last_24hrs
one_row_per_cow_per_day.total_lifetime_yield
one_row_per_cow_per_day.lmsfd
one_row_per_cow_per_day.lmsft
one_row_per_cow_per_day.lmsv
one_row_per_cow_per_day.bcsfd
one_row_per_cow_per_day.bcsft
one_row_per_cow_per_day.bcsv
one_row_per_cow_per_day.allflex_tag_number
one_row_per_cow_per_day.avg_health_index_for_non_milked_cows
one_row_per_cow_per_day.min_health_index_for_non_milked_cows
one_row_per_cow_per_day.max_health_index_for_non_milked_cows
one_row_per_cow_per_day.sum_health_index_without_milk_for_test
one_row_per_cow_per_day.avg_health_index_without_milk_for_test
one_row_per_cow_per_day.min_health_index_without_milk_for_test
one_row_per_cow_per_day.max_health_index_without_milk_for_test
one_row_per_cow_per_day.sum_scr_heat_index
one_row_per_cow_per_day.avg_scr_heat_index
one_row_per_cow_per_day.min_scr_heat_index
one_row_per_cow_per_day.max_scr_heat_index
one_row_per_cow_per_day.avg_activity_change
one_row_per_cow_per_day.min_activity_change
one_row_per_cow_per_day.max_activity_change
one_row_per_cow_per_day.avg_activity_change_by_2_hours
one_row_per_cow_per_day.min_activity_change_by_2_hours
one_row_per_cow_per_day.max_activity_change_by_2_hours
one_row_per_cow_per_day.daily_activity
one_row_per_cow_per_day.avg_raw_activity_data
one_row_per_cow_per_day.min_raw_activity_data
one_row_per_cow_per_day.max_raw_activity_data
one_row_per_cow_per_day.daily_rumination
one_row_per_cow_per_day.avg_raw_rumination
one_row_per_cow_per_day.min_raw_rumination
one_row_per_cow_per_day.max_raw_rumination
one_row_per_cow_per_day.weekly_rumination_average
one_row_per_cow_per_day.daily_weight
one_row_per_cow_per_day.percent_deviation_of_daily_weight_from_weekly_average_weight
one_row_per_cow_per_day.yesterday_s_weight
one_row_per_cow_per_day.avg_temp
one_row_per_cow_per_day.min_temp
one_row_per_cow_per_day.max_temp
one_row_per_cow_per_day.sum_act
one_row_per_cow_per_day.avg_act
one_row_per_cow_per_day.min_act
one_row_per_cow_per_day.max_act
one_row_per_cow_per_day.avg_act_index
one_row_per_cow_per_day.min_act_index
one_row_per_cow_per_day.max_act_index
one_row_per_cow_per_day.sum_drink_cycles_v2
one_row_per_cow_per_day.avg_temp_without_drink_cycles
one_row_per_cow_per_day.min_temp_without_drink_cycles
one_row_per_cow_per_day.max_temp_without_drink_cycles
one_row_per_cow_per_day.avg_temp_normal_index
one_row_per_cow_per_day.min_temp_normal_index
one_row_per_cow_per_day.max_temp_normal_index
one_row_per_cow_perDay.sum_smx_heat_index
one_row_per_cow_perDay.avg_smx_heat_index
one_row_per_cow_perDay.min_smx_heat_index
one_row_per_cow_perDay.max_smx_heat_index
one_row_per_cow_perDay.sum_calving_index
one_row_per_cow_perDay.avg_calving_index
one_row_per_cow_perDay.min_calving_index
one_row_per_cow_perDay.max_calving_index
one_row_per_cow_perDay.avg_rum_index
one_row_per_cow_perDay.min_rum_index
one_row_per_cow_perDay.max_rum_index
one_row_per_cow_perDay.sum_water_intake
one_row_per_cow_perDay.penCount
one_row_per_cow_perDay.totalDM
one_row_per_cow_perDay.totalDMPerHead
one_row_per_cow_perDay.totalAF
one_row_per_cow_perDay.totalAFPerHead
one_row_per_cow_perDay.refusalAmount
one_row_per_cow_perDay.refusalPct
one_row_per_cow_perDay.targetDMPerHeadGrams
one_row_per_cow_perDay.recipeNames
one_row_per_cow_perDay.totalCost
one_row_per_cow_perDay.costPerHead
one_row_per_cow_perDay.env_temp_1
one_row_per_cow_perDay.env_rh_1
"""

class AgentState(TypedDict):
    question: str
    sql_query: Optional[str]
    rows: Optional[list[dict]]
    data_insights: Optional[str]
    error: Optional[str]
    
#node1: generate SQL from question
def generate_sql_node(state: AgentState) -> AgentState:
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    prompt = f"""
    You are a SQLite SQL generator.

    Return ONLY SQL and USE THE TABLE NAME AND SCHEMA I GAVE YOU. 
    THIS IS THE TABLE NAME aggregated_data.one_row_per_cow_per_day USE THIS
    Return ONLY the columns asked for USING THE COLUMN NAMES I GAVE U IN THE SCHEMA.

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
        db_url = "postgresql+psycopg://postgres:farmdata2024@localhost:5433/FotF"
        engine = create_engine(db_url)
        
        # pandas read_sql_query handles the execution and fetching
        df = pd.read_sql_query(state["sql_query"], engine)         
        
        # Convert to list of dicts for your next node
        rows = df.head(100).to_dict(orient="records") 
        return {**state, "rows": rows, "error": None}
        
     except Exception as e:       
        return {**state, "rows": None, "error": f"SQL Error: {str(e)}"}
        

def analyze_data(state: AgentState) -> AgentState:
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    prompt = f"""
    You are a agent that takes SQL results and translates them into natural language.

    Return ONLY the translation of the SQL results.
    Return ONLY natural language.
    Return ONLY the insights that can be drawn from the data.
    Do NOT return the raw data, just the insights.

    Schema:
    {SCHEMA}

    Question:
    {state['question']} 

    SQL Results:
    {state['rows']}
    """

    try: 
        response = llm.invoke(([HumanMessage(content=prompt)]))
        response_clean = response.content.strip()
        return {**state, "data_insights": response_clean, "error": None} #copies the dictionary and edits the data_insights and error keys
    except Exception as e:
        return {**state, "data_insights": None, "error": f"Error generating insights: {str(e)}"}


def build_graph():
     graph = StateGraph(AgentState)
     graph.add_node("generate_sql", generate_sql_node)
     graph.add_node("execute_sql", execute_sql_node)
     graph.add_node("analyze_data", analyze_data)


     graph.add_edge(START, "generate_sql")
     graph.add_edge("generate_sql", "execute_sql")
     graph.add_edge("execute_sql", "analyze_data")
     graph.add_edge("analyze_data", END)

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