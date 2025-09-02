from langgraph.graph import MessagesState


class BotState(MessagesState):
    init_message: str
    user_query: str
    clarification_message: str
    search_query: str