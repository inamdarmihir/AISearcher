# Research and Summarization Agent

This project implements a research and summarization agent using LangGraph, OpenAI GPT-4o, Streamlit, Weaviate vector database for RAG, and Exa for search.

## System Architecture

The system includes the following components:

1. **Router Agent** – Determines whether a query should be answered using the LLM, web research, or retrieval-augmented generation (RAG).
2. **Web Research Agent** – Performs web search using Exa API and extracts relevant information.
3. **RAG Agent** – Retrieves information from a Weaviate vector database.
4. **Summarization Agent** – Synthesizes information from all sources into a comprehensive response.

All agents are orchestrated using LangGraph with LangSmith tracing enabled.

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install langchain langchain-openai langchain-community langchain-core langgraph langsmith streamlit weaviate-client exa-py python-dotenv
```

3. Set up environment variables in `.env` file:

```
OPENAI_API_KEY=your_openai_api_key
EXA_API_KEY=your_exa_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=research_agent
WEAVIATE_URL=your_weaviate_url
WEAVIATE_API_KEY=your_weaviate_api_key
```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

## Project Structure

- `router_agent.py` - Implements the router agent for query classification
- `web_research_agent.py` - Implements the web research agent using Exa API
- `rag_agent.py` - Implements the RAG agent using Weaviate
- `summarization_agent.py` - Implements the summarization agent
- `agent_graph.py` - Integrates all agents using LangGraph
- `app.py` - Streamlit interface
- `test_router_agent.py` - Unit tests for the router agent

## Features

- **Intelligent Query Routing**: Automatically determines the best approach for answering queries
- **Web Research**: Searches the web for up-to-date information using Exa API
- **Knowledge Base Retrieval**: Retrieves relevant information from a vector database
- **Comprehensive Summarization**: Synthesizes information from multiple sources
- **Clean UI**: Professional Streamlit interface with query history
- **Tracing and Debugging**: LangSmith integration for monitoring agent behavior

## Requirements

- Python 3.10+
- OpenAI API key
- Exa API key
- LangSmith API key (for tracing)
- Weaviate instance (can use embedded version for testing)

## License

MIT
