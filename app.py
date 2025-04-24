import os
import streamlit as st
import time
from typing import Dict, Any
import traceback
from dotenv import load_dotenv

# Import our agent graph
from agent_graph import run_agent

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Research & Summarization Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner, more professional look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .agent-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .error-message {
        background-color: #ffebee;
        color: #c62828;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-message {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-message {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .routing-info {
        font-style: italic;
        color: #546e7a;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

if 'processing' not in st.session_state:
    st.session_state.processing = False

# Title and description
st.title("🔍 Research & Summarization Agent")
st.markdown("""
<div class="info-message">
This agent processes your queries by determining whether they require reasoning from an LLM, 
web research, or retrieval from a knowledge base. It leverages multiple specialized sub-agents 
to generate well-structured responses.
</div>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This research and summarization agent uses:
    - **LangGraph** for agent orchestration
    - **OpenAI GPT-4o** for reasoning and summarization
    - **Exa API** for web search
    - **Qdrant** for vector database RAG
    - **LangSmith** for tracing and debugging
    
    The system includes:
    1. **Router Agent** – Determines the best approach for answering your query
    2. **Web Research Agent** – Searches the web for up-to-date information
    3. **RAG Agent** – Retrieves information from a knowledge base
    4. **Summarization Agent** – Synthesizes information into a comprehensive response
    """)
    
    st.header("Sample Queries")
    st.markdown("""
    - What is quantum computing?
    - What are the latest developments in artificial intelligence?
    - Explain the theory of relativity
    - What is the current state of climate change?
    """)

# Main content area
query = st.text_area("Enter your query:", height=100, key="query_input")

# Query processing
col1, col2 = st.columns([1, 5])
with col1:
    submit_button = st.button("Submit", use_container_width=True)
with col2:
    if submit_button or st.session_state.processing:
        if not query and not st.session_state.processing:
            st.error("Please enter a query.")
        elif not st.session_state.processing:
            # Start processing
            st.session_state.processing = True
            st.rerun()

# Process the query
if st.session_state.processing and query:
    try:
        # Display processing message
        with st.status("Processing your query...", expanded=True) as status:
            st.write("Analyzing query and determining the best approach...")
            
            # Run the agent
            start_time = time.time()
            result = run_agent(query)
            end_time = time.time()
            
            # Extract information from result
            route = result.get("route", "unknown")
            reasoning = result.get("reasoning", "")
            error = result.get("error", "")
            final_answer = result.get("final_answer", "")
            
            # Update status based on route
            if route == "llm":
                st.write("Using LLM to answer your query...")
            elif route == "web_research":
                st.write("Performing web research to find the latest information...")
            elif route == "rag":
                st.write("Retrieving information from knowledge base...")
            
            st.write("Synthesizing information into a comprehensive response...")
            status.update(label="Query processed!", state="complete")
        
        # Display the result
        st.markdown("## Results")
        
        # Display routing information
        st.markdown(f"""
        <div class="routing-info">
        Query routed to: <strong>{route.upper()}</strong><br>
        Reasoning: {reasoning}<br>
        Processing time: {end_time - start_time:.2f} seconds
        </div>
        """, unsafe_allow_html=True)
        
        # Display any errors
        if error:
            st.markdown(f"""
            <div class="error-message">
            Error encountered: {error}
            </div>
            """, unsafe_allow_html=True)
        
        # Display the final answer
        st.markdown(final_answer)
        
        # Add to history
        st.session_state.history.append({
            "query": query,
            "route": route,
            "answer": final_answer,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Reset processing state
        st.session_state.processing = False
        
    except Exception as e:
        # Handle exceptions
        st.markdown(f"""
        <div class="error-message">
        <strong>An error occurred:</strong><br>
        {str(e)}<br><br>
        {traceback.format_exc()}
        </div>
        """, unsafe_allow_html=True)
        
        # Reset processing state
        st.session_state.processing = False

# Display history
if st.session_state.history:
    st.markdown("## Query History")
    
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Query: {item['query'][:50]}... ({item['time']})"):
            st.markdown(f"**Route:** {item['route'].upper()}")
            st.markdown(item['answer'])

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; color: #666;">
Powered by LangGraph, OpenAI GPT-4o, Exa, and Qdrant<br>
© 2025 Research & Summarization Agent
</div>
""", unsafe_allow_html=True)
