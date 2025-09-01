from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv('config.env')
assert os.getenv('OPENAI_API_KEY') is not None
assert os.getenv('TAVILY_API_KEY') is not None

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
)
