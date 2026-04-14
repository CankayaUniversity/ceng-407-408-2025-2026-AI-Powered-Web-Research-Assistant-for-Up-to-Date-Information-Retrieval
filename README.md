# Autonomous Web Research Agent (v1.0 - Core Engine)

This project is a local research assistant built with FastAPI, LangGraph, and Llama 3.1 via Ollama.  
It can search the web using Tavily and read specific pages in depth using BeautifulSoup.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed locally
- Llama model downloaded in Ollama:

```bash
ollama pull llama3.1
```

## Project Structure

- `main.py`: FastAPI entrypoint (`app`)
- `api.py`: API routes and request handling
- `agent.py`: LangGraph ReAct agent builder
- `tools.py`: Tavily + deep page reader tools
- `config.py`: shared configuration and system prompt

## Setup

1. Clone the project and open the root directory.
2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate it:

- PowerShell: `.\venv\Scripts\Activate.ps1`
- CMD: `.\venv\Scripts\activate.bat`
- macOS/Linux: `source venv/bin/activate`

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Add your Tavily API key into `.env`:

```env
TAVILY_API_KEY=tvly-...
```

## Run

Start Ollama in one terminal:

```bash
ollama serve
```

Start the API in another terminal:

```bash
uvicorn main:app --reload
```

Open Swagger UI:

- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## API Usage

- `GET /`  
  Health/info endpoint.

- `GET /ask_agent`  
  Query parameter: `question`  
  Example:
  - `question=What is LangGraph?`

Example response:

```json
{
  "your_question": "What is LangGraph?",
  "agent_answer": "..."
}
```

If an error occurs:

```json
{
  "error": "Agent error: ..."
}
```
