import unittest
from unittest.mock import patch, MagicMock

# Import our agents
from router_agent import route_query

class TestRouterAgent(unittest.TestCase):
    """
    Test cases for the Router Agent component.
    """
    
    @patch('router_agent.router_chain')
    def test_router_agent(self, mock_router_chain):
        """Test the router agent's ability to route queries correctly."""
        # Mock the router chain responses
        mock_router_chain.invoke.side_effect = [
            {"route": "llm", "reasoning": "This is a general knowledge question"},
            {"route": "web_research", "reasoning": "This requires current information"},
            {"route": "rag", "reasoning": "This is about a specific topic in our knowledge base"}
        ]
        
        # Test LLM routing
        llm_query = "Explain the theory of relativity"
        llm_result = route_query(llm_query)
        self.assertEqual(llm_result["route"], "llm")
        
        # Test web research routing
        web_query = "What are the latest developments in quantum computing?"
        web_result = route_query(web_query)
        self.assertEqual(web_result["route"], "web_research")
        
        # Test RAG routing
        rag_query = "What is quantum computing?"
        rag_result = route_query(rag_query)
        self.assertEqual(rag_result["route"], "rag")
    
    @patch('router_agent.router_chain')
    def test_router_error_handling(self, mock_router_chain):
        """Test the router agent's error handling."""
        # Mock an error in the router chain
        mock_router_chain.invoke.side_effect = Exception("Test error")
        
        # Test error handling
        query = "Test query"
        result = route_query(query)
        
        # Verify default to LLM on error
        self.assertEqual(result["query"], query)
        self.assertEqual(result["route"], "llm")
        self.assertIn("Error occurred during routing", result["reasoning"])

if __name__ == "__main__":
    unittest.main()
