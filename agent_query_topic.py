from functools import lru_cache

from langchain_openai import ChatOpenAI
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool, BaseTool
from dotenv import load_dotenv
from bot_state import BotState
from utils import print_messages
from typing import Dict
from tavily import TavilyClient
from mlflow.entities import SpanType
import mlflow
import logging

logger = logging.getLogger(__name__)

system_prompt = """
You are an AI assistant that helps patients learn about health-related topics and medical conditions. 
You guide users by recognizing relevant medical questions, retrieving accurate information, 
and explaining it in a way that’s clear, supportive, and easy for non-experts to understand.

You have access to the following tools to support your tasks.
## Available Tools
* tool_search_query(keywords: str)
  Use this tool when the user's question is clearly health-related. 
  Convert the question into concise search keywords that help retrieve information about the topic’s definition, 
  causes, symptoms, and treatment.
* tool_clarification(message: str)
  Use this tool when the user's input is unclear or not related to health. Provide a polite clarification prompt. 
  This tool will return the user's clarified input as a string, which you must treat as a new query and re-evaluate.
  
## Your Responsibilities
1. Interpret the user’s query
  Determine if the input is clearly about a health topic, symptom, condition, treatment, or medical process.
2. If the query is health-related:
  * Rephrase it into concise search keywords (e.g., “diabetes causes symptoms treatment”).
  * Use the tool tool_search_query(keywords) to retrieve information.
  * Once the result is returned, write a clear, friendly, and accurate summary covering:
    * What the condition or topic is
    * Its causes or risk factors
    * Common symptoms
    * Treatments or management options
3. If the query is not clearly health-related or is vague:
  * Use tool_clarification(message) to ask the user (politely) to clarify or rephrase their question with a health-related focus.
  * Once the clarified message is returned, process it as if it were a new user input — go back to step 1.
    
## Examples of Appropriate Inputs for tool_search_query
* "COVID" → "COVID-19 overview symptoms treatment prevention"
* “What causes asthma?” → "asthma causes symptoms treatment"
* “Tell me about high cholesterol” → "high cholesterol explanation causes symptoms treatment"
* “I want to learn about depression” → "depression causes symptoms treatment"

## Examples Needing tool_clarification
* “What’s the best laptop?”
* “Tell me more”
* “I want to learn something”
Use a response like:
“Could you please clarify what health topic or medical condition you’d like to learn about?”

## Tone
* Always be respectful, supportive, and clear.
* Avoid medical jargon; use simple explanations.
* Encourage curiosity and self-learning.

## Important Notes
* You must use the available tools to respond: either tool_search_query() or tool_clarification().
* If you receive a clarified input from the tool_clarification() tool, treat it exactly as if the user just asked it.
* Do not fabricate information — only summarize after retrieving valid content from tool_search_query.
"""

# Load configurations
load_dotenv('config.env')
assert os.getenv('OPENAI_API_KEY') is not None
assert os.getenv('TAVILY_API_KEY') is not None


@tool
@mlflow.trace(span_type=SpanType.TOOL)
def tool_clarification(message: str) -> str:
    """
    This tool would prompt the message to the user and ask the user to clarify the query.
    :param message: The message to ask the user to clarify
    :return: The user's response
    """
    user_clarification = input(message + "\n")
    return user_clarification


@tool
@mlflow.trace(span_type=SpanType.TOOL)
def tool_search_query(keywords: str) -> Dict:
    """
    This tool would search the passed in keywords using the Tavily client
    :param keywords: The keywords to search for
    :return: The search results
    """
    logger.debug(f"Start Tavily searching with keywords: {keywords}")
    tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
    response = tavily_client.search(keywords)
    return response


tools = [tool_clarification, tool_search_query]

@lru_cache(maxsize=1)
def create_llm():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
    )

    llm = llm.bind_tools(tools)
    return llm


def node_init_user_query(state: BotState) -> BotState:
    """
    Prompt a message to the user and get the user's initial query message
    """
    user_query = state["user_query"]
    messages = [
        SystemMessage(system_prompt),
        HumanMessage(user_query)
    ]
    return {
        "messages": messages,
        "user_query": user_query,
    }


def node_health_agent(state: BotState) -> BotState:
    """
    Let AI validate whether the user's input is health-related.
    Returns the AI response which indicates whether needs clarification or should start search
    """
    messages = state["messages"]
    llm = create_llm()
    ai_message = llm.invoke(messages)
    logger.debug(f"AI message: {ai_message}")
    return {"messages": ai_message}


def node_tools(state: BotState) -> BotState:
    # print_messages(state["messages"])
    result = []

    for tool_call in state["messages"][-1].tool_calls:
        # tool = tools_by_name[tool_call["name"]]
        # ret = tool.invoke({**tool_call["args"]})
        ret = None
        if tool_call["name"] == "tool_clarification":
            ret = tool_clarification({**tool_call["args"]})
            state["clarification_message"] = tool_call["args"]["message"]
            state["user_query"] = ret
        if tool_call["name"] == "tool_search_query":
            ret = tool_search_query({**tool_call["args"]})
            state["search_query"] = tool_call["args"]["keywords"]
        result.append(
            ToolMessage(content=ret, tool_call_id=tool_call["id"])
        )
    state["messages"].extend(result)
    return state


def route_clarify_or_search(state: BotState):
    """
    Check the AI response to decide whether needs clarification or can start to search
    """
    # print_messages(state["messages"])
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return node_tools.__name__
    return node_summary.__name__


def node_summary(state: BotState) -> BotState:
    last_message = state["messages"][-1]
    # last_message.pretty_print()
    return {"search_summary": last_message.content}
