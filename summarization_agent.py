import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Initialize the OpenAI model
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)

class SummarizationAgent:
    """
    Agent for synthesizing and summarizing information from multiple sources.
    """
    
    def __init__(self):
        """
        Initialize the summarization agent.
        """
        # Create the summarization prompt
        self.summarization_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a summarization agent that synthesizes information from multiple sources into a coherent, well-structured response.
            
Your task is to analyze the provided information and create a comprehensive summary that directly addresses the user's query.
The summary should be well-organized, factual, and include proper citations to the original sources.

Follow these guidelines:
1. Start with a concise overview that directly answers the query
2. Organize the information into logical sections with clear headings
3. Highlight key points and important details
4. Include relevant facts, figures, and examples
5. Cite sources for specific information
6. Conclude with a brief summary of the main findings
7. Add a "Sources" section at the end listing all references

Aim for clarity, accuracy, and completeness in your summary.
"""),
            ("human", """
Query: {query}

Information from LLM:
{llm_info}

Information from Web Research:
{web_info}

Information from Knowledge Base:
{kb_info}

Synthesize this information into a comprehensive, well-structured response.
""")
        ])
        
        # Create the summarization chain
        self.summarization_chain = self.summarization_prompt | model | StrOutputParser()
    
    def summarize(self, 
                 query: str, 
                 llm_info: str = "", 
                 web_info: Dict[str, Any] = None, 
                 kb_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Summarize information from multiple sources.
        
        Args:
            query: The original user query
            llm_info: Information from direct LLM response
            web_info: Information from web research
            kb_info: Information from knowledge base
            
        Returns:
            Dictionary containing the summary and metadata
        """
        try:
            # Extract web research information
            web_research_text = ""
            web_sources = []
            
            if web_info and web_info.get("success", False):
                web_research_text = web_info.get("information", "")
                web_sources = web_info.get("sources", [])
            
            # Extract knowledge base information
            kb_text = ""
            
            if kb_info and kb_info.get("success", False):
                kb_text = kb_info.get("answer", "")
            
            # Generate summary
            summary = self.summarization_chain.invoke({
                "query": query,
                "llm_info": llm_info or "No direct LLM information available.",
                "web_info": web_research_text or "No web research information available.",
                "kb_info": kb_text or "No knowledge base information available."
            })
            
            # Collect all sources
            sources = web_sources
            
            return {
                "query": query,
                "summary": summary,
                "sources": sources
            }
        except Exception as e:
            print(f"Error in summarization: {e}")
            return {
                "query": query,
                "summary": f"Failed to generate summary: {str(e)}",
                "sources": []
            }
    
    def format_output(self, result: Dict[str, Any]) -> str:
        """
        Format the summarization result for display.
        
        Args:
            result: The summarization result
            
        Returns:
            Formatted output as a string
        """
        output = f"# Response to: {result['query']}\n\n"
        output += result['summary']
        
        # Add sources if available
        if result['sources']:
            output += "\n\n## Sources\n"
            for i, source in enumerate(result['sources'], 1):
                output += f"{i}. [{source.get('title', 'Untitled')}]({source.get('url', '#')})\n"
        
        return output

# Test the summarization agent
if __name__ == "__main__":
    agent = SummarizationAgent()
    
    # Sample data
    query = "Explain quantum computing and its recent developments"
    
    llm_info = "Quantum computing is a type of computation that harnesses quantum mechanical phenomena. It uses quantum bits or qubits which can exist in multiple states simultaneously due to superposition."
    
    web_info = {
        "success": True,
        "information": "Recent developments in quantum computing include Google's claim of quantum supremacy in 2019, IBM's 127-qubit processor in 2021, and advances in error correction techniques. Companies like Microsoft, Amazon, and Intel are also investing heavily in quantum research.",
        "sources": [
            {"title": "Quantum Computing Advances", "url": "https://example.com/quantum1"},
            {"title": "Latest in Quantum Technology", "url": "https://example.com/quantum2"}
        ]
    }
    
    kb_info = {
        "success": True,
        "answer": "Quantum computers use quantum bits or qubits. Unlike classical bits, qubits can exist in a state of superposition, allowing them to be in multiple states simultaneously. This property, along with quantum entanglement, gives quantum computers the potential to solve certain problems much faster than classical computers."
    }
    
    result = agent.summarize(query, llm_info, web_info, kb_info)
    formatted_output = agent.format_output(result)
    
    print(formatted_output)
