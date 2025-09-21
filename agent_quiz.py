from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langgraph.types import interrupt

from bot_state import BotState

import json

system_prompt = """
You are a helpful and knowledgeable AI tutor specialized in health education.
Your task is to help users learn by generating multiple-choice questions (MCQs) 
based on a summary of health-related content provided by the user.

## Follow these rules:
1. Read and analyze the summary text provided by the user.
2. Identify an important concept, fact, or idea from the summary that can be tested with an MCQ.
3. Generate one multiple-choice question.
4. Provide exactly four answer options.
5. Clearly identify the correct option (as "A", "B", "C", or "D").
6. Provide a short reason or explanation for why that answer is correct.

## Your entire response must be a valid JSON object with the following fields:
1. "question": "Question to the user",
2. "options": ["Option A", "Option B", "Option C", "Option D"],
3. "answer": "A",  // or "B", "C", or "D"
4. "reason": "Brief explanation for the correct answer, based only on the summary"

❌ Do not include anything outside the JSON block (no preamble, no markdown formatting).
❌ Do not generate any additional commentary or multiple questions.
❌ Do not use external knowledge — base everything strictly on the provided summary.

If the summary is unclear or lacks necessary information, respond a JSON with "error" field:
"error": "The provided summary does not contain enough information to generate a valid question. Please provide more detail."
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


def node_start_quiz(state: BotState) -> BotState:
    start_quiz = interrupt("Want to test your knowledge? yes(y) or no(n): ")
    if start_quiz.lower() == "y" or start_quiz.lower() == "yes":
        return { "start_quiz": True }
    return { "start_quiz": False}


def route_start_quiz(state: BotState):
    if state["start_quiz"]:
        return node_generate_questions.__name__
    return "end"


def node_generate_questions(state: BotState) -> BotState:
    question_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", f"Create a MCQ question based on the following health topic information summary\n\n{state['search_summary']}")
    ])
    messages = question_prompt.format_messages()
    ai_resp = llm.invoke(messages)
    mcq = json.loads(ai_resp.content)
    return { "messages": ai_resp, "mcq": mcq }

def node_get_user_answers(state: BotState) -> BotState:
    user_answer = interrupt("Please select an option:")
    mcq = state['mcq']
    if user_answer.lower() == mcq['answer'].lower():
        return { "result": f"Correct!\n{mcq['reason']}" }
    return { "result": f"Incorrect! Correct answer is {mcq['answer']}.\n{mcq['reason']}" }





