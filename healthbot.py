from bot_state import BotState
from agent_query_topic import (node_init_user_query, node_health_agent, route_clarify_or_search, node_tools,
                               node_summary)
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

import mlflow
import logging

logging.basicConfig(filename="healthbot.log", level=logging.DEBUG)
logger = logging.getLogger(__name__)
tracking_uri = "http://127.0.0.1:5123"
mlflow.set_tracking_uri(tracking_uri)
experiment = mlflow.set_experiment("healthbot")

workflow = StateGraph(BotState)

# Add agent nodes
workflow.add_node(node_init_user_query.__name__, node_init_user_query)
workflow.add_node(node_health_agent.__name__, node_health_agent)
workflow.add_node(node_tools.__name__, node_tools)
workflow.add_node(node_summary.__name__, node_summary)

# Add edges
workflow.add_edge(START, node_init_user_query.__name__)
workflow.add_edge(node_init_user_query.__name__, node_health_agent.__name__)
workflow.add_conditional_edges(
    node_health_agent.__name__,
    route_clarify_or_search,
    {
        node_tools.__name__: node_tools.__name__,
        node_summary.__name__: node_summary.__name__
    }
)
workflow.add_edge(node_tools.__name__, node_health_agent.__name__)
workflow.add_edge(node_summary.__name__, END)

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# from utils import display_graph
# display_graph(graph)

mlflow.langchain.autolog()

init_state = {"init_message": "What health topic or medical condition would you like to learn about?\n"}
config = {"configurable": {"thread_id": "1"}}
final_state = graph.invoke(init_state, config)
print("===== Summary of the information =====")
print(final_state["search_summary"])
