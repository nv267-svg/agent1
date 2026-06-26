import pandas as pd
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from sqlalchemy import create_engine
import os
import pandas as pd
from .build_features import build_cow_lactation_features
from inference_example import predict
import json


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.2:3b-instruct-fp16")


 
class AgentState(TypedDict):
    question: str
    animal_id: Optional[str]
    lact: Optional[str]
    df_raw: Optional[pd.DataFrame]
    features_df: Optional[pd.DataFrame]
    prediction: Optional[dict]
    answer: Optional[str]
    error: Optional[str]


#node 1: get the animal id and lactation from ollama 
def extract_features_node(state: AgentState) -> AgentState:
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        format="json",  # Ollama JSON mode -> response.content is always valid JSON
    )
 
    prompt = f"""
Read the question and return a JSON object with these keys:
 
- animal_id: the cow's identification number, as a string. null if not mentioned.
- lact: the lactation number, as a string. null if not mentioned.
- dim_min: integer start of the "days in milk" window if one is mentioned, else 1
- dim_max: integer end of the "days in milk" window if one is mentioned, else 21
 
Question:
{state['question']}
"""
 
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        data = json.loads(response.content)
 
        animal_id = data.get("animal_id")
        lact = data.get("lact")
 
        if not animal_id or not lact:
            return {
                **state,
                "animal_id": animal_id,
                "lact": lact,
                "error": "Couldn't find both a cow ID and a lactation number in the question.",
            }
 
        return {
            **state,
            "animal_id": str(animal_id),
            "lact": str(lact),
            "error": None,
        }
    except Exception as e:
        return {**state, "error": f"Error extracting cow/lactation info: {str(e)}"}




#node 2: get the data frame from the postgres databse (same code as old agent2graph.py)

def fetch_data_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return state
 
    try:
        db_url = "postgresql+psycopg://postgres:farmdata2024@128.84.40.194:5432/FotF"
        engine = create_engine(db_url)
 
        df_raw = pd.read_sql(f"""
            SELECT * FROM aggregated_data.one_row_per_cow_per_day
            WHERE animal_id = '{state["animal_id"]}' AND lact = '{state["lact"]}'
            AND dim BETWEEN 1 AND 21
        """, engine)
 
    except Exception as e:
        print(f"Error connecting to database: {e}")
        df_raw = pd.DataFrame()  # Empty DataFrame if connection fails
 
    if df_raw.empty:
        return {
            **state,
            "df_raw": df_raw,
            "error": (
                f"No rows found for animal_id={state['animal_id']}, "
                f"lact={state['lact']}, dim 1-21."
            ),
        }
 
    return {**state, "df_raw": df_raw, "error": None}


# node 3: turn rows into features for the model 
def build_features_node(state: AgentState) -> AgentState:
    if state.get("error") or state.get("df_raw") is None:
        return state
 
    try:
        # min_early_records=0 so a single sparse cow doesn't get silently dropped
        features_df = build_cow_lactation_features(state["df_raw"], min_early_records=0)
        return {**state, "features_df": features_df, "error": None}
    except Exception as e:
        return {**state, "features_df": None, "error": f"Feature-building error: {str(e)}"}

# node 4: predict raw values  
def predict_node(state: AgentState) -> AgentState:
    if state.get("error") or state.get("features_df") is None:
        return state
 
    try:
        result = predict(state["features_df"])
        return {**state, "prediction": result, "error": None}
    except Exception as e:
        return {**state, "prediction": None, "error": f"Prediction error: {str(e)}"}

# node 5: analyze the prediction in natural language 
def analyze_node(state: AgentState) -> AgentState:
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
 
    if state.get("error"):
        prompt = f"""
    Explain this problem to a dairy farm manager in one short, plain sentence.
        Do not mention code, Python, SQL, or stack traces.
 
    Problem: {state['error']}
    """
    else:
        prompt = f"""
       You are an assistant explaining a dairy cow model prediction to a farm
    manager who is not technical.
 
        Return BOTH the natural language version and the raw data version. 
 
    Question:
    {state['question']}
 
    Cow: animal_id={state.get('animal_id')}, lactation={state.get('lact')}, 
    days in milk 1-21
 
    Model output:
    {state.get('prediction')}
    """
 
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return {**state, "answer": response.content.strip()}
    except Exception as e:
        return {**state, "answer": None, "error": f"Error generating explanation: {str(e)}"}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("extract_features", extract_features_node)
    graph.add_node("fetch_data", fetch_data_node)
    graph.add_node("build_features", build_features_node)
    graph.add_node("predict", predict_node)
    graph.add_node("analyze", analyze_node)
 
    graph.add_edge(START, "extract_features")
    graph.add_edge("extract_features", "fetch_data")
    graph.add_edge("fetch_data", "build_features")
    graph.add_edge("build_features", "predict")
    graph.add_edge("predict", "analyze")
    graph.add_edge("analyze", END)
 
    return graph.compile()

cow_prediction_agent = build_graph()


if __name__ == "__main__":
    out = cow_prediction_agent.invoke(
        {"question": "What's the prediction for cow 2075 in lactation 6?"}
    )
    if out.get("error"):
        print("Error:", out["error"])
    print("\nAnswer:")
    print(out["answer"])
