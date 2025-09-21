from langgraph.types import Command

from agent_quiz import node_start_quiz, route_start_quiz, node_generate_questions, node_get_user_answers
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
    workflow.add_node(node_start_quiz.__name__, node_start_quiz)
    workflow.add_node(node_generate_questions.__name__, node_generate_questions)
    workflow.add_node(node_get_user_answers.__name__, node_get_user_answers)

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
    workflow.add_edge(node_summary.__name__, node_start_quiz.__name__)
    workflow.add_conditional_edges(
        node_start_quiz.__name__,
        route_start_quiz,
        {
            node_generate_questions.__name__: node_generate_questions.__name__,
            "end": END
        }
    )
    workflow.add_edge(node_generate_questions.__name__, node_get_user_answers.__name__)
    workflow.add_edge(node_get_user_answers.__name__, END)

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
        # Wait for the summary and print
        for event in graph.stream(init_state, config, stream_mode="updates"):
            if 'node_summary' in event:
                print("===== Summary of the information =====")
                print(event['node_summary']["search_summary"])

        # Get the user select on whether to start the quiz
        start_quiz = input("Want to test your knowledge? yes(y) or no(n): ")
        while start_quiz.lower() not in ["y", "yes", "n", "no"]:
            start_quiz = input("Want to test your knowledge? yes(y) or no(n): ")

        if start_quiz.lower() != "y":
            return

        for event in graph.stream(Command(resume=start_quiz), config, stream_mode="updates"):
            if 'node_generate_questions' in event:
                mcq = event['node_generate_questions']['mcq']
                print(mcq["question"])
                for option in mcq["options"]:
                    print(option)

        user_answer = input("Please select the correct answer (A/B/C/D):\n")
        while user_answer.lower() not in ["a", "b", "c", "d"]:
            user_answer = input("Please select the correct answer (A/B/C/D):\n")
        for event in graph.stream(Command(resume=user_answer), config, stream_mode="updates"):
            if 'node_get_user_answers' in event:
                print(event['node_get_user_answers']['result'])

    except Exception as e:
        print(f"Error: {e}")
        logger.error(traceback.format_exc())
        traceback.print_exc()


if __name__ == "__main__":
    main()