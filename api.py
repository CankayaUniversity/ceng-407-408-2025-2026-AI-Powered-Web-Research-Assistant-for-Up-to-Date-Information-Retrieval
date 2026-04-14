from fastapi import FastAPI
from langchain_core.messages import HumanMessage

from agent import build_agent
from config import load_environment
from fact_extraction import extract_claims

TOOL_LOG_PREVIEW_CHARS = 2000


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
                tool_output = str(message.content)
                print(f"\n[TOOL RESULT ({message.name})]:")
                if len(tool_output) <= TOOL_LOG_PREVIEW_CHARS:
                    print(tool_output)
                else:
                    print(tool_output[:TOOL_LOG_PREVIEW_CHARS])
                    print(f"... (truncated, total chars: {len(tool_output)})")
        print("=" * 40 + "\n")

        final_answer = result["messages"][-1].content.replace("\n", " ")
        tool_messages = [
            {"name": message.name, "content": str(message.content)}
            for message in result["messages"]
            if message.type == "tool"
        ]
        extraction = extract_claims(final_answer, tool_messages)

        print("FACT EXTRACTION SUMMARY")
        for idx, fact in enumerate(extraction["facts"], start=1):
            print(f" FACT {idx}: {fact['claim_text']}")
            print(f"  URLs: {fact['evidence_urls']}")
            print(f"  FLAGS: {fact['fact_quality_flags']}")
        print(f" TRUST SIGNALS: {extraction['trust_signals']}")

        return {
            "your_question": question,
            "agent_answer": final_answer,
            "facts": extraction["facts"],
            "sources": extraction["sources"],
            "trust_signals": extraction["trust_signals"],
        }
    except Exception as e:
        return {"error": f"Agent error: {str(e)}"}
