from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from config import MODEL_NAME, MODEL_TEMPERATURE, SYSTEM_MESSAGE
from tools import build_tools


def build_agent():
    llm = ChatOllama(model=MODEL_NAME, temperature=MODEL_TEMPERATURE)
    tools = build_tools()
    return create_react_agent(llm, tools, prompt=SYSTEM_MESSAGE)