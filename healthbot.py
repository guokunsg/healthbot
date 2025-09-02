from bot_state import BotState
from agent_query_topic import node_init_user_query, node_validate_user_query, clarify_or_search, node_clarify_query, node_search_query
from langgraph.graph import StateGraph, START, END


workflow = StateGraph(BotState)

# Add agent nodes
workflow.add_node(node_init_user_query.__name__, node_init_user_query)
workflow.add_node(node_validate_user_query.__name__, node_validate_user_query)
workflow.add_node(node_clarify_query.__name__, node_clarify_query)
workflow.add_node(node_search_query.__name__, node_search_query)

# Add edges
workflow.add_edge(START, node_init_user_query.__name__)
workflow.add_edge(node_init_user_query.__name__, node_validate_user_query.__name__)
workflow.add_conditional_edges(
    node_validate_user_query.__name__,
    clarify_or_search,
    {
        node_clarify_query.__name__: node_clarify_query.__name__,
        node_search_query.__name__: node_search_query.__name__
    }
)
workflow.add_edge(node_clarify_query.__name__, node_validate_user_query.__name__)
workflow.add_edge(node_search_query.__name__, END)

graph = workflow.compile()

# display_graph(graph)

input = { "init_message": "What health topic or medical condition would you like to learn about?\n" }
for s in graph.stream(input):
    print(s)
