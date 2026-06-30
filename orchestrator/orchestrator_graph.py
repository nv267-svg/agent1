import os
import json
import requests
import uuid
import pandas as pd
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from sqlalchemy import create_engine
import pandas as pd

from graph import crop_agent, AGENT_CARD as CROP_AGENT_CARD
from agent2.agent2graph import cow_prediction_agent, COW_AGENT_CARD

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.2:3b-instruct-fp16")
COW_AGENT_URL = os.getenv("COW_AGENT_URL", "http://cow-exit-prediction-agent.team1.svc.cluster.local:8080")
CROP_AGENT_URL = os.getenv("CROP_AGENT_URL", "http://crop-agent.team1.svc.cluster.local:8080")

class OrchestratorState(TypedDict):
    question: str
    chosen_agent: Optional[str] #Crop or cow
    answer: Optional[str]
    error: Optional[str]

#Format a call into JSON for the orchestrator agent to call the cow or crop agent
#uses HTTP POST to send the question to the agent and get the answer back
def call_agent(agent_url: str, question: str) -> str:
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "parts": [{ "kind": "text", "text": question }]
            }
        }
    }
    response = requests.post(agent_url, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
          raise RuntimeError(data["error"]["message"])
    return data["result"]["parts"][0]["text"]


#Node 1
def choose_agent_node(state: OrchestratorState) -> OrchestratorState:
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        format="json",  
    )

    crop_skills = CROP_AGENT_CARD["skills"][0] #access the skill in the agent card
    cow_skills = COW_AGENT_CARD["skills"][0]
 
    #accesses the agent card directly in case we change it
    prompt = f"""
    You are a router agent. Given a user's question, decide which agent should answer it.

    Agent "crop":
    Description: {CROP_AGENT_CARD['description']}
    Skill: {crop_skills['name']} : {crop_skills['description']}
    Example questions: {crop_skills['examples']}

    Agent "cow":
    Description: {COW_AGENT_CARD['description']}
    Skill: {cow_skills['name']} : {cow_skills['description']}
    Example questions: {cow_skills['examples']}

    Return a JSON object with one key:
    - chosen_agent: either "crop" or "cow", based on which agent is best suited to answer the question.
    
    Question:
    {state['question']}
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        data = json.loads(response.content)
        chosen_agent = data.get("chosen_agent")
        if chosen_agent not in ("crop", "cow"):
            return {**state, "error": f"Could not determine target agent (got: {chosen_agent})"}

        return {**state, "chosen_agent": chosen_agent, "error": None}
    except Exception as e:
        return {**state, "error": f"Error classifying question: {str(e)}"}
    
    #node 2a: call the crop agent
def call_crop_agent_node(state: OrchestratorState) -> OrchestratorState:
    if state.get("error"):
        return state  # Skip if there's already an error
    try: 
        answer = call_agent(CROP_AGENT_URL, state["question"])
        return {**state, "answer": answer, "error": None}
    except Exception as e:
        return {**state, "error": f"Error calling crop agent: {str(e)}"}
    
    #node 2b: call the cow agent
def call_cow_agent_node(state: OrchestratorState) -> OrchestratorState:
    if state.get("error"):
        return state  # Skip if there's already an error
    try: 
        answer = call_agent(COW_AGENT_URL, state["question"])
        return {**state, "answer": answer, "error": None}
    except Exception as e:
        return {**state, "error": f"Error calling cow agent: {str(e)}"}

#this is a choosing helper function for the graph which routes to either cow or crop agent based on the chosen agent; t
def route_to_agent(state: OrchestratorState) -> str:
    if state.get("error"):
        return "end"
    return state["chosen_agent"]

def build_graph():
    graph = StateGraph(OrchestratorState)
    graph.add_node("choose_agent", choose_agent_node), 
    graph.add_node("call_crop", call_crop_agent_node),
    graph.add_node("call_cow", call_cow_agent_node),
        
    graph.add_edge(START, "choose_agent"),
    #uses the routing function to determine which agent to call
    graph.add_conditional_edges(
        "choose_agent",
         route_to_agent,
        {
            "crop": "call_crop",
            "cow": "call_cow",
            "end": END,
        }, 
    )
    graph.add_edge("call_crop", END),
    graph.add_edge("call_cow", END)

    return graph.compile()

orchestrator_graph = build_graph()

#test
"""
if __name__ == "__main__":
    out = orchestrator_graph.invoke({
        "question": "What's the prediction for cow 2075 in lactation 6?",
        "target_agent": None,
        "answer": None,
        "error": None,
    })
    if out.get("error"):
        print("Error:", out["error"])
    else:
        print("Routed to:", out["target_agent"])
        print("Answer:", out["answer"])
"""