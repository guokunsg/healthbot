from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


def display_graph(graph):
    image_bytes = graph.get_graph().draw_mermaid_png()
    # img = Image(graph.get_graph().draw_mermaid_png())
    with open("graph.png", "wb") as png:
        png.write(image_bytes)

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