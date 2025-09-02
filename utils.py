from IPython.display import Image, display
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

def display_graph(graph):
    display(
        Image(
            graph.get_graph().draw_mermaid_png()
        )
    )

def print_messages(messages):
    print("--- Number of messages:", len(messages), "---")
    for m in messages:
        if isinstance(m, HumanMessage):
            m.pretty_print()
        elif isinstance(m, AIMessage):
            m.pretty_print()
        else:
            print("===== Ignored message with class: ", m.__class__.__name__, "=====")
    print("------------")