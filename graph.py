from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from sqlalchemy import create_engine
import os
import pandas as pd
import sqlite3
import requests
import psycopg

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.2:3b-instruct-fp16")
#DB_PATH         = os.getenv("DB_PATH",         "crop.db")

SCHEMA = """
Table: aggregated_data.one_row_per_cow_per_day

Column names: 
One_row_per_cow_per_day.date = the date the data was collected
One_row_per_cow_per_day.animal_id = the identification number of the cow
One_row_per_cow_per_day.lact =  the lactation number of the cow. 
One_row_per_cow_per_day.dim = days in milk. 
One_row_per_cow_per_day.pen = what PEN number the cow is in
One_row_per_cow_per_day.cbrd = stands for cattle breed 
One_row_per_cow_per_day.pta =  Predicted Transmitting Ability (need more info)
One_row_per_cow_per_day.gdpr =  genomic prediction of daughter pregnancy rate. 
One_row_per_cow_per_day.fdat =  fresh date, means the date the cow began her current lactation. 
One_row_per_cow_per_day.cdat = conception date
One_row_per_cow_per_day.ddat =  dry date
One_row_per_cow_per_day.rc  =   reproductive code numbered(0-7)RC=3 OK/OPEN Animals that are eligible to breed. These could be fresh animals that were checked and declared ready to breed or a bred animal that was declared open. RC=4 BRED Inseminated animals that have not been diagnosed as PREG or OPEN. RC=5 PREG Animals that have been declared pregnant. RC=6 DRY Animals in their Dry period (not milking). RC=7 SLD/DIE Any animal that was sold or has died. 
One_row_per_cow_per_day.rpro  = reproductive code in written language 
One_row_per_cow_per_day.dcc  = days_carried_calf if calf pregnant
One_row_per_cow_per_day.scc somatic_cell_count. Raw somatic cell counts in exact value where 0-3 
One_row_per_cow_per_day.ddry = days dry which is the number of days that a cow is not being milked 
One_row_per_cow_per_day.psirc previous lactation sire of conception which tracks the sire 
One_row_per_cow_per_day.tbrd = the number of times the cow has been inseminated during the CURRENT REPRODUCTIVE CYCLE. 
One_row_per_cow_per_day.milkings_yesterday = how many times the cow was milked yesterday
one_row_per_cow_per_day.milk_yield_yesterday
one_row_per_cow_per_day.yield_yesterday_session_1
one_row_per_cow_per_day.yield_yesterday_session_2
one_row_per_cow_per_day.yield_yesterday_session_3
one_row_per_cow_per_day.avg_daily_yield_last_7d
one_row_per_cow_per_day.yesterday_deviation_pct_from_avg_last_7d
one_row_per_cow_per_day.total_yield_in_lactation
one_row_per_cow_per_day.milk_yield_last_24hrs
one_row_per_cow_per_day.total_lifetime_yield
One_row_per_cow_per_day.lmsfd = Date the last locomotion score was added
One_row_per_cow_per_day.lmsft = Time of day the last locomotion score was added 
One_row_per_cow_per_day.lmsv = Last locomotion score value to indicate mobility. Range is 0 to 100
One_row_per_cow_per_day.bcsfd = Date the last BCS was added
One_row_per_cow_per_day.bcsft = Time of day the last BCS was added
One_row_per_cow_per_day.bcsv = The last BCS Value. BCS = “Body Condition Score” The value indicates an animal’s energy balance based on being extremely thin or having excessive fat deposits. Range is 0 to 5
One_row_per_cow_per_day.allflex_tag_number = the visual identification number printed on an Allflex tag attached to the cow.
one_row_per_cow_per_day.avg_health_index_for_non_milked_cows
one_row_per_cow_per_day.min_health_index_for_non_milked_cows
one_row_per_cow_per_day.max_health_index_for_non_milked_cows
one_row_per_cow_per_day.sum_health_index_without_milk_for_test
one_row_per_cow_per_day.avg_health_index_without_milk_for_test
one_row_per_cow_per_day.min_health_index_without_milk_for_test
one_row_per_cow_per_day.max_health_index_without_milk_for_test
One_row_per_cow_per_day.sum_scr_heat_index 
One_row_per_cow_per_day.avg_scr_heat_index		
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
One_row_per_cow_per_day.daily_rumination = time spent per day chewing cud, or ruminating. 
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
    If I say for the first 5 cows for example, that just means the first 5 listed you do not need to concern yourself
    with the animal_id field. The actual column label is the part BEFORE the equal sign the part AFTER is a description to help you understand what the label means. DO NOT assume what the labels mean read the part in the schema

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
        db_url = "postgresql+psycopg://postgres:farmdata2024@128.84.40.194:5432/FotF"
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