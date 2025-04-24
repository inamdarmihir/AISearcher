import os
from typing import TypedDict, Literal, Dict, Any, List, Annotated, Union
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import langgraph.graph as lg
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langsmith import Client

# Import our agents
from router_agent import route_query, RouterState
from web_research_agent import WebResearchAgent
from rag_agent import RAGAgent
from summarization_agent import SummarizationAgent

# Load environment variables
load_dotenv()

# Initialize LangSmith client for tracing
client = Client()

# Initialize the OpenAI model
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)

# Initialize our agents
web_research_agent = WebResearchAgent(
    max_results=8,
    concurrent_processing=True,
    enable_pii_detection=True,
    enable_guardrails=True,
    structured_output=True,
    privacy_level="high"  # Set to high to block all privacy-sensitive queries
)
rag_agent = RAGAgent()
summarization_agent = SummarizationAgent()

# Define the agent state
class AgentState(TypedDict):
    query: str
    route: Literal["llm", "web_research", "rag", "hybrid", None]
    reasoning: str
    entities: List[str]
    llm_info: str
    web_info: Dict[str, Any]
    kb_info: Dict[str, Any]
    final_answer: str
    error: str

# Define the initial state
def create_initial_state(query: str) -> AgentState:
    """
    Create the initial state for the agent.
    
    Args:
        query: The user query
        
    Returns:
        Initial state
    """
    return {
        "query": query,
        "route": None,
        "reasoning": "",
        "entities": [],
        "llm_info": "",
        "web_info": {},
        "kb_info": {},
        "final_answer": "",
        "error": ""
    }

# Define the router node
def router_node(state: AgentState) -> AgentState:
    """
    Route the query to the appropriate agent.
    
    Args:
        state: The current state
        
    Returns:
        Updated state with routing decision
    """
    try:
        result = route_query(state["query"])
        
        return {
            **state,
            "route": result["route"],
            "reasoning": result["reasoning"],
            "entities": result.get("entities", [])
        }
    except Exception as e:
        return {
            **state,
            "error": f"Error in router: {str(e)}",
            "route": "llm"  # Default to LLM if there's an error
        }

# Define the LLM node
def llm_node(state: AgentState) -> AgentState:
    """
    Process the query using direct LLM.
    
    Args:
        state: The current state
        
    Returns:
        Updated state with LLM response
    """
    try:
        # Create a simple prompt for the LLM
        messages = [
            HumanMessage(content=f"Please answer the following question: {state['query']}")
        ]
        
        # Get response from LLM
        response = model.invoke(messages)
        
        return {
            **state,
            "llm_info": response.content
        }
    except Exception as e:
        return {
            **state,
            "error": f"Error in LLM processing: {str(e)}",
            "llm_info": f"Failed to get LLM response: {str(e)}"
        }

# Define the web research node
def web_research_node(state: AgentState) -> AgentState:
    """
    Process the query using web research.
    
    Args:
        state: The current state
        
    Returns:
        Updated state with web research results
    """
    try:
        # Perform web research
        result = web_research_agent.research(state["query"])
        
        return {
            **state,
            "web_info": result
        }
    except Exception as e:
        return {
            **state,
            "error": f"Error in web research: {str(e)}",
            "web_info": {
                "query": state["query"],
                "success": False,
                "error": str(e),
                "information": None,
                "sources": []
            }
        }

# Define the RAG node
def rag_node(state: AgentState) -> AgentState:
    """
    Process the query using RAG.
    
    Args:
        state: The current state
        
    Returns:
        Updated state with RAG results
    """
    try:
        # Add sample data if not already added
        # In a real application, this would be done during initialization
        # or through a separate data ingestion process
        rag_agent.add_sample_data()
        
        # Query the knowledge base
        result = rag_agent.query(state["query"])
        
        return {
            **state,
            "kb_info": result
        }
    except Exception as e:
        return {
            **state,
            "error": f"Error in RAG: {str(e)}",
            "kb_info": {
                "query": state["query"],
                "success": False,
                "error": str(e),
                "answer": None,
                "documents": []
            }
        }

# Define the hybrid node that combines RAG with another approach
def hybrid_node(state: AgentState) -> AgentState:
    """
    Process the query using both RAG and another approach (LLM or web_research).
    
    Args:
        state: The current state
        
    Returns:
        Updated state with hybrid results
    """
    try:
        # First perform RAG to get knowledge base information
        rag_state = rag_node(state)
        
        # Check if RAG was successful
        if rag_state["kb_info"].get("success", False):
            # If RAG found relevant information, also get LLM insights
            llm_state = llm_node(state)
            
            # Combine the results
            return {
                **state,
                "kb_info": rag_state["kb_info"],
                "llm_info": llm_state["llm_info"],
                "error": rag_state.get("error", "") + " " + llm_state.get("error", "")
            }
        else:
            # If RAG didn't find good results, try web research instead
            web_state = web_research_node(state)
            
            # Combine the results (even if RAG wasn't successful, we include its info for completeness)
            return {
                **state,
                "kb_info": rag_state["kb_info"],
                "web_info": web_state["web_info"],
                "error": rag_state.get("error", "") + " " + web_state.get("error", "")
            }
    except Exception as e:
        return {
            **state,
            "error": f"Error in hybrid processing: {str(e)}",
            "kb_info": state.get("kb_info", {})
        }

# Define the summarization node
def summarization_node(state: AgentState) -> AgentState:
    """
    Summarize the information from all sources.
    
    Args:
        state: The current state
        
    Returns:
        Updated state with final answer
    """
    try:
        # Summarize the information
        result = summarization_agent.summarize(
            query=state["query"],
            llm_info=state["llm_info"],
            web_info=state["web_info"],
            kb_info=state["kb_info"]
        )
        
        # Format the output
        formatted_output = summarization_agent.format_output(result)
        
        return {
            **state,
            "final_answer": formatted_output
        }
    except Exception as e:
        return {
            **state,
            "error": f"Error in summarization: {str(e)}",
            "final_answer": f"Failed to generate summary: {str(e)}"
        }

# Define the routing logic
def decide_route(state: AgentState) -> Literal["llm", "web_research", "rag", "hybrid", "summarize"]:
    """
    Decide which node to route to based on the state.
    
    Args:
        state: The current state
        
    Returns:
        The next node to route to
    """
    # If there's an error, go straight to summarization
    if state["error"]:
        return "summarize"
    
    # Route based on the router's decision
    return state["route"]

# Create the graph
def create_agent_graph() -> StateGraph:
    """
    Create the agent graph.
    
    Returns:
        The agent graph
    """
    # Create a new graph
    graph = StateGraph(AgentState)
    
    # Add the nodes
    graph.add_node("router", router_node)
    graph.add_node("llm", llm_node)
    graph.add_node("web_research", web_research_node)
    graph.add_node("rag", rag_node)
    graph.add_node("hybrid", hybrid_node)
    graph.add_node("summarize", summarization_node)
    
    # Set the entry point
    graph.set_entry_point("router")
    
    # Add the edges
    graph.add_conditional_edges(
        "router",
        decide_route,
        {
            "llm": "llm",
            "web_research": "web_research",
            "rag": "rag",
            "hybrid": "hybrid"
        }
    )
    
    # Connect all processing nodes to summarization
    graph.add_edge("llm", "summarize")
    graph.add_edge("web_research", "summarize")
    graph.add_edge("rag", "summarize")
    graph.add_edge("hybrid", "summarize")
    
    # Set the exit point
    graph.add_edge("summarize", END)
    
    # Compile the graph
    return graph.compile()

# Create a function to run the agent
def run_agent(query: str) -> Dict[str, Any]:
    """
    Run the agent on a query.
    
    Args:
        query: The user query
        
    Returns:
        The final state
    """
    # Create the initial state
    initial_state = create_initial_state(query)
    
    # Create the graph
    graph = create_agent_graph()
    
    # Create a memory saver for checkpoints
    memory_saver = MemorySaver()
    
    # Run the graph with tracing
    result = graph.invoke(
        initial_state,
        {
            "configurable": {
                "thread_id": "research_agent",
                "run_name": f"Query: {query[:50]}...",
                "project_name": os.getenv("LANGCHAIN_PROJECT", "research_agent")
            },
            "checkpoint_saver": memory_saver
        }
    )
    
    return result

# Test the integrated agent
if __name__ == "__main__":
    # Test queries
    test_queries = [
        "What is quantum computing?",
        "What are the latest developments in artificial intelligence?",
        "Explain the theory of relativity",
        "What is the current state of climate change?",
        "What is fluid analytics.ai?",
        "What is CRISPR technology?",
    ]
    
    for query in test_queries:
        print(f"\nProcessing query: {query}")
        result = run_agent(query)
        
        print("\nFinal Answer:")
        print(result["final_answer"])
        
        if result["error"]:
            print(f"\nError: {result['error']}")
        
        print("-" * 80)
