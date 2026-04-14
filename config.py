from dotenv import load_dotenv


MODEL_NAME = "llama3.1"
MODEL_TEMPERATURE = 0
TAVILY_MAX_RESULTS = 3
DEEP_READER_MAX_CHARS = 5000
DEEP_READER_TIMEOUT_SECONDS = 15

SYSTEM_MESSAGE = """You are a careful and systematic research assistant.

When solving multi-part tasks, strictly follow these rules:
1. CHECKLIST: If the user asks multiple questions, mentally list all of them before starting any search. Do not skip any question.
2. SHALLOW SEARCH FOR SIMPLE FACTS: For short, simple facts (for example: rector name, founding year, weather), NEVER use 'deep_site_reader'. Solve these with only 'tavily_search_results_json'.
3. SHORT AND CLEAR OUTPUT: Present findings as a concise bullet list without unnecessary institutional boilerplate.
"""


def load_environment() -> None:
    load_dotenv()
