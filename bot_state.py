from typing import Any

from langgraph.graph import MessagesState


class BotState(MessagesState):
    # Initial prompt message to get the user query
    init_message: str
    # User's input query
    user_query: str
    # Extra clarification message for the user. User's input will be stored into user_query
    clarification_message: str
    # The search keywords
    search_query: str
    # The summary of the search result
    search_summary: str
    # Whether the user selected to start the quiz
    start_quiz: bool
    # MCQ questions and answers from the AI
    mcq: dict[str, Any]
    # The result message
    result: str