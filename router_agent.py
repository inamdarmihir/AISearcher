import os
from typing import Literal, TypedDict, Annotated, List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import json
import re

# Load environment variables
load_dotenv()

# Define the router agent state
class RouterState(TypedDict):
    query: str
    route: Literal["llm", "web_research", "rag", "hybrid", None]
    reasoning: str
    entities: Optional[List[str]]

# Initialize the OpenAI model with strict JSON mode
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    response_format={"type": "json_object"}  # Force JSON output
)

# Create a simpler entity extraction prompt without template issues
entity_extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an entity extraction agent. Your job is to identify any specific entities in the query that might need to be researched or looked up. 
    
Entities include:
- Specific companies, products, or services
- Technical terms or concepts that might be new or specialized
- Names of people, places, or organizations
- Any potentially unfamiliar terminology
- Locations, landmarks, or geographic places

Output ONLY a JSON array of the identified entities in the exact format:
{{"entities": ["entity1", "entity2"]}}

If no notable entities are found, return an empty array: {{"entities": []}}
"""),
    ("human", "Query: {query}")
])

entity_extraction_chain = entity_extraction_prompt | model | StrOutputParser()

# Create the router prompt with improved instructions and without template issues
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a router agent that determines how to process user queries.
    
Your job is to analyze the query and decide on the best routing approach:

1. "llm" - Answered directly using an LLM (for general knowledge, reasoning, or opinion questions that don't require specific or up-to-date information)

2. "web_research" - Researched on the web (for current events, specific facts, time-sensitive information, questions about specific companies, products, services, or real-world locations)

3. "rag" - Retrieved from a knowledge base (for established domain knowledge or concepts likely in our knowledge base)

4. "hybrid" - A combination of RAG and LLM or RAG and web_research (for queries that need both retrieval from a knowledge base and enhancement with general reasoning or web research)

ROUTING GUIDELINES:
- ALWAYS use "web_research" for:
  * Specific companies, products, or services
  * Real-world locations or places (restaurants, landmarks, addresses)
  * Questions asking "where is X" for any real entity
  * Current events, news, or recent developments
  * Data that changes or updates over time

- Use "hybrid" for:
  * Technical concepts that might need both factual retrieval and explanation
  * Complex topics that benefit from both knowledge base lookup and supplementary information

- Use "rag" for:
  * Well-established academic topics
  * Historical information likely in our knowledge base
  * Standard scientific or technical concepts

- Use "llm" only for:
  * Pure opinion or reasoning questions
  * Hypothetical scenarios
  * Simple general knowledge with no specific entities

YOU MUST RESPOND WITH VALID JSON ONLY in this exact format:
{{"route": "web_research", "reasoning": "This is a question about a specific location"}}

DO NOT include any explanations outside of the JSON.
"""),
    ("human", "Query: {query}\nEntities: {entities_str}")
])

# First get the raw output for parsing
router_raw_chain = router_prompt | model | StrOutputParser()

# Function to extract entities from a query
def extract_entities(query: str) -> List[str]:
    """
    Extract potentially unfamiliar entities from the query.
    
    Args:
        query: The user query
        
    Returns:
        List of extracted entities
    """
    try:
        raw_output = entity_extraction_chain.invoke({"query": query})
        
        # Try to parse the JSON response
        try:
            # Extract JSON if it's wrapped
            json_str = raw_output.strip()
            
            # Parse the JSON
            result = json.loads(json_str)
            
            if "entities" in result and isinstance(result["entities"], list):
                return result["entities"]
            else:
                return []
        except Exception as e:
            print(f"Error parsing entity extraction output: {e}")
            # Fallback: try to extract entities using regex
            potential_entities = re.findall(r'"([^"]+)"', raw_output)
            if potential_entities:
                return potential_entities
            return []
    except Exception as e:
        print(f"Error in entity extraction: {e}")
        return []

# Function to route the query
def route_query(query: str) -> RouterState:
    """
    Route the query to the appropriate agent based on the query content.
    
    Args:
        query: The user query
        
    Returns:
        RouterState: The state with routing decision
    """
    # Pre-processing check for location queries
    query_lower = query.lower()
    
    # First check for obvious location or place queries before any complex processing
    if ("where is" in query_lower or 
        "location of" in query_lower or
        "address of" in query_lower or
        "directions to" in query_lower or
        "how to get to" in query_lower or
        "find" in query_lower and any(x in query_lower for x in ["restaurant", "cafe", "shop", "store", "mall", "place"])):
        
        return {
            "query": query,
            "route": "web_research",
            "reasoning": "This is a location-based query that requires real-world geographic information.",
            "entities": []  # We'll extract entities later if needed
        }
    
    # Extract entities from the query for other cases
    entities = extract_entities(query)
    print(f"Extracted entities: {entities}")
    
    try:
        # Create a string representation of entities for the prompt template
        entities_str = ", ".join(entities) if entities else "None"
        
        # Get raw output from the model
        raw_output = router_raw_chain.invoke({
            "query": query,
            "entities_str": entities_str
        })
        
        # Parse the JSON response (with forced JSON mode, this should be clean)
        try:
            result = json.loads(raw_output)
            
            # Validate the result
            if "route" not in result or "reasoning" not in result:
                raise ValueError("Missing required fields in JSON response")
                
            if result["route"] not in ["llm", "web_research", "rag", "hybrid"]:
                raise ValueError(f"Invalid route: {result['route']}")
                
            return {
                "query": query,
                "route": result["route"],
                "reasoning": result["reasoning"],
                "entities": entities
            }
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing model output: {e}")
            print(f"Raw output: {raw_output}")
            
            # Apply special rules based on extracted entities and query patterns
            return apply_fallback_routing(query, entities, str(e))
    except Exception as e:
        print(f"Error in router agent: {e}")
        return apply_fallback_routing(query, entities, str(e))

def apply_fallback_routing(query: str, entities: List[str], error_msg: str) -> RouterState:
    """
    Apply fallback routing rules when the primary routing fails.
    
    Args:
        query: The user query
        entities: Extracted entities from the query
        error_msg: The error message from the failed routing attempt
        
    Returns:
        RouterState: The fallback routing decision
    """
    query_lower = query.lower()
    
    # Location-based query detection
    location_terms = ["where", "located", "location", "address", "find", "place", "restaurant", "hotel", 
                     "shop", "store", "building", "street", "road", "avenue", "directions", "map"]
    
    if any(term in query_lower for term in location_terms):
        return {
            "query": query,
            "route": "web_research",
            "reasoning": f"Query appears to be asking about a physical location. Routing to web research.",
            "entities": entities
        }
    
    # Check for company or product indicators
    if any(keyword in query_lower for keyword in [".ai", ".com", ".org", ".net", "company", "product", "service", "platform", "app", "website"]):
        return {
            "query": query,
            "route": "web_research",
            "reasoning": f"Query appears to be about a specific company or product. Routing to web research.",
            "entities": entities
        }
    
    # Check if there are extracted entities that seem like specialized terms
    if entities and any(entity.lower() not in query_lower.replace(entity.lower(), "") for entity in entities):
        return {
            "query": query,
            "route": "hybrid",
            "reasoning": f"Query contains potentially specialized entities: {entities}. Using hybrid approach.",
            "entities": entities
        }
    
    # Check for current events, news, or recent developments indicators
    if any(keyword in query_lower for keyword in ["latest", "recent", "current", "news", "today", "update", "development"]):
        return {
            "query": query,
            "route": "web_research",
            "reasoning": f"Query appears to request current information. Routing to web research.",
            "entities": entities
        }
    
    # Check for academic or established knowledge indicators
    if any(keyword in query_lower for keyword in ["theory", "concept", "principle", "formula", "define", "explain", "what is", "how does"]):
        return {
            "query": query,
            "route": "hybrid",
            "reasoning": f"Query appears to be about conceptual knowledge. Using hybrid approach.",
            "entities": entities
        }
    
    # Default to LLM for other queries
    return {
        "query": query,
        "route": "llm",
        "reasoning": f"Default fallback due to error: {error_msg}",
        "entities": entities
    }

# Test the router
if __name__ == "__main__":
    test_queries = [
        "What is the capital of France?",
        "What are the latest developments in quantum computing?",
        "Explain the theory of relativity",
        "Who won the last presidential election?",
        "What is the current price of Bitcoin?",
        "What is fluid analytics.ai?",
        "Tell me about Microsoft Corporation",
        "What services does AWS offer?",
        "What is CRISPR technology?",
        "How does neuromorphic computing work?",
        "What is the Metaverse?",
        "Where is Wadeshwar in Kothrud?",
        "How do I get to Central Park in New York?",
        "What's the address of the Taj Mahal?",
    ]
    
    for query in test_queries:
        result = route_query(query)
        print(f"Query: {query}")
        print(f"Entities: {result.get('entities', [])}")
        print(f"Route: {result['route']}")
        print(f"Reasoning: {result['reasoning']}")
        print("-" * 50)
