from bot_state import BotState
from agent_query_topic import (node_init_user_query, node_health_agent, route_clarify_or_search, node_tools,
                               node_summary)
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

import mlflow
import logging
import traceback


def create_health_bot_graph():
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

    return graph


def main():
    try:
        logging.basicConfig(filename="healthbot.log", level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        tracking_uri = "http://127.0.0.1:5123"
        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.set_experiment("healthbot")

        graph = create_health_bot_graph()
        mlflow.langchain.autolog()

        # from utils import display_graph
        # display_graph(graph)

        user_message = "What health topic or medical condition would you like to learn about?\n"
        user_query = input(user_message)
        init_state = { "user_query": user_query }
        config = {"configurable": {"thread_id": "1"}}
        final_state = graph.invoke(init_state, config)
        # for message in final_state["messages"]:
        #     print(f"Message: {message}")
        print("===== Summary of the information =====")
        print(final_state["search_summary"])
    except Exception as e:
        print(f"Error: {e}")
        logger.error(traceback.format_exc())
        traceback.print_exc()


if __name__ == "__main__":
    main()