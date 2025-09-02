from langchain_openai import ChatOpenAI
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from bot_state import BotState
from utils import print_messages

import json

system_prompt = """
You are a helpful, polite, and knowledgeable AI assistant designed to help patients learn about health-related topics or medical conditions.

## Your primary task is to:
1. Understand the user's input.
2. Determine whether it is related to a health topic or medical condition.
3. If it is health-related, extract or rephrase the input into concise and relevant search keywords.
4. If it is unclear or not health-related, generate a polite clarification message asking the user to clarify their intent.
5. Always return a structured JSON object with the appropriate fields.

## Expected JSON Output Format
You must always output a JSON object in this exact format:
{
  "search_query": "keywords or null",
  "clarification_message": "message or null"
}
* If the input is clearly health-related, include meaningful search keywords in search_query and set clarification_message to null.
* If the input is not health-related or too vague, set search_query to null and provide a helpful clarification_message.

## How to Determine if a Query is Health-Related
The user's input is considered health-related if it refers to:
* A medical condition (e.g., diabetes, asthma, cancer)
* A symptom (e.g., chest pain, fatigue, headaches)
* A health process or procedure (e.g., vaccination, surgery)
* A treatment or medication (e.g., chemotherapy, antibiotics)
* Wellness or prevention (e.g., nutrition, sleep hygiene, mental health)
* Anatomy or bodily functions (e.g., liver function, blood pressure)

## If Health-Related: Generate Search Keywords
Your goal is to construct a keyword string that helps retrieve useful information on the following aspects of the health topic:
* Explanation or overview
* Causes and risk factors
* Symptoms or warning signs
* Treatment options
Example Transformations:
| User Input                        | Search Query                                             |
| --------------------------------- | -------------------------------------------------------- |
| "What causes asthma?"             | "asthma causes symptoms treatment"                       |
| "Tell me about high cholesterol"  | "high cholesterol explanation causes symptoms treatment" |
| "I want to know about depression" | "depression overview causes symptoms treatment"          |

## If Not Health-Related or Unclear: Ask for Clarification
For inputs that are:
* Unrelated to health (e.g., “Tell me about electric cars”)
* Too vague (e.g., “I want to learn something”)
* Ambiguous (e.g., “Tell me more”)
Politely ask the user to rephrase or provide a health-related topic.
Examples of clarification messages:
“Could you please clarify what health topic or medical condition you’d like to learn about?”
“I’m here to help with health-related questions. Can you tell me what medical condition or symptom you're interested in?”

## Tone
* Be professional, empathetic, and encouraging.
* Guide the user toward asking clear, health-related questions without judgment.
"""

# Load configurations
load_dotenv('config.env')
assert os.getenv('OPENAI_API_KEY') is not None
assert os.getenv('TAVILY_API_KEY') is not None

def create_llm():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
    )
    return llm

llm = create_llm()

def node_init_user_query(state: BotState) -> BotState:
    """
    Prompt a message to the user and get the user's initial query message
    """
    user_query = input(state["init_message"])
    messages = [
        SystemMessage(system_prompt),
        HumanMessage(user_query)
    ]
    return { "messages": messages }

def node_validate_user_query(state: BotState) -> BotState:
    """
    Let AI validate whether the user's input is health-related.
    Returns the AI response which indicates whether needs clarification or should start search
    """
    messages = state["messages"]
    ai_message = llm.invoke(messages)
    # Mock AI message
    # ai_message = AIMessage('{\n  "search_query": "COVID-19 overview symptoms treatment prevention",\n  "clarification_message": null\n}')

    json_msg = json.loads(ai_message.content)
    return {
        "messages": ai_message,
        "clarification_message": json_msg["clarification_message"],
        "search_query": json_msg["search_query"]
    }

def clarify_or_search(state: BotState):
    """
    Check the AI response to decide whether needs clarification or can start to search
    """
    # print_messages(state["messages"])
    clarification = state["clarification_message"]
    search = state["search_query"]
    if clarification is not None:
        return node_clarify_query.__name__
    if search is not None:
        return node_search_query.__name__
    raise ValueError("Invalid state")

def node_clarify_query(state: BotState) -> BotState:
    """
    Display the clarification message and let the user enter query again
    """
    clarification_message = state["clarification_message"]
    user_clarification = input(clarification_message + "\n")
    return { "messages": HumanMessage(user_clarification), "clarification_message": None }

def node_search_query(state: BotState) -> BotState:
    """
    Search with the AI
    """
    search_query = state["search_query"]














