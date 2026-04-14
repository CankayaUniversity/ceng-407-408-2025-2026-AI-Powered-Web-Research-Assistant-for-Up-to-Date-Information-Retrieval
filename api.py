from fastapi import FastAPI
from langchain_core.messages import HumanMessage

from agent import build_agent
from config import load_environment


load_environment()
app = FastAPI()
autonomous_agent = build_agent()


@app.get("/")
def home():
    return {"message": "Deep research agent is active!"}


@app.get("/ask_agent")
def ask_agent(question: str):
    try:
        input_message = {"messages": [HumanMessage(content=question)]}
        result = autonomous_agent.invoke(input_message)

        print("\n" + "=" * 40)
        print("AGENT REASONING TRACE")
        for message in result["messages"]:
            if message.type == "ai" and hasattr(message, "tool_calls") and message.tool_calls:
                print("\n[THOUGHT]: The agent decided to use tools.")
                for tool_call in message.tool_calls:
                    print(f" TOOL USED: {tool_call['name']}")
                    print(f" TOOL INPUT: {tool_call['args']}")
            elif message.type == "tool":
                print(f"\n[TOOL RESULT ({message.name})]: {message.content[:150]}... (truncated)")
        print("=" * 40 + "\n")

        final_answer = result["messages"][-1].content.replace("\n", " ")
        return {"your_question": question, "agent_answer": final_answer}
    except Exception as e:
        return {"error": f"Agent error: {str(e)}"}
