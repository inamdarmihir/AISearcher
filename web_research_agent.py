import os
import time
import traceback
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from exa_py import Exa
from concurrent.futures import ThreadPoolExecutor, as_completed
import langchain.globals as langchain_globals  # Import for configuring LangChain caching

# Load environment variables
load_dotenv()

# Configure LangChain caching
langchain_globals.set_llm_cache(None)  # Disable caching to fix the error

# Initialize Exa client with error handling
try:
    exa_client = Exa(api_key=os.getenv("EXA_API_KEY"))
    print("Successfully initialized Exa client")
except Exception as e:
    print(f"Error initializing Exa client: {str(e)}")
    exa_client = None

# Initialize the OpenAI model (removed caching due to LangChain error)
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)

class WebResearchAgent:
    """
    Agent for performing web research using Exa API.
    """
    
    def __init__(self, max_results: int = 8, max_retries: int = 3, concurrent_processing: bool = True):
        """
        Initialize the web research agent.
        
        Args:
            max_results: Maximum number of search results to retrieve
            max_retries: Maximum number of retries for API calls
            concurrent_processing: Whether to process results concurrently
        """
        self.max_results = max_results
        self.max_retries = max_retries
        self.concurrent_processing = concurrent_processing
        
        # Create the extraction prompt with improved instructions for deeper analysis
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert web research agent that performs comprehensive analysis and extraction of information from search results.
            
Your task is to analyze the provided search results and extract ALL relevant information related to the user's query.
Focus on providing a thorough, detailed response that covers:

1. Key facts and statistics
2. Recent developments and timeline of events
3. Different perspectives and viewpoints
4. Expert opinions and analysis
5. Contextual background information essential to understanding the topic
6. Connections between related aspects of the topic

When researching events:
- Provide specific dates, locations, and involved parties
- Explain causes and consequences
- Include casualty figures and damage assessments if applicable
- Mention official responses and ongoing investigations

For information about organizations or companies:
- Include founding details, leadership, and market position
- Describe key products, services, and unique offerings
- Detail recent partnerships, acquisitions, or major announcements
- Explain their approach to technology and innovation
- Discuss market strategy and competitive advantages

For information about specific products or technologies:
- Describe technical specifications and capabilities
- Compare with competing products or technologies
- Mention pricing, availability, and target markets when relevant
- Discuss real-world applications and use cases
- Note recent updates, developments, or controversies

If the search results don't contain relevant information, clearly state what specific information is missing 
and suggest what more specialized sources might have better information on this topic.

Synthesize the information into a coherent, well-organized response. Use clear headings to separate different aspects.
ALWAYS cite your sources for each piece of information, using the format: [Source Title: URL].
"""),
            ("human", """
Query: {query}

Search Results:
{search_results}

Provide a comprehensive analysis that directly answers the query with in-depth, detailed information.
""")
        ])
        
        # Create the extraction chain
        self.extraction_chain = self.extraction_prompt | model | StrOutputParser()
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform a comprehensive web search using Exa API.
        
        Args:
            query: The search query
            
        Returns:
            List of search results
        """
        if exa_client is None:
            print("Exa client not initialized, cannot perform search")
            return []
        
        start_time = time.time()
        print(f"Starting search with query: '{query}'")
            
        for attempt in range(self.max_retries):
            try:
                print(f"Attempting Exa search (attempt {attempt+1}/{self.max_retries})")
                
                # First try to get more recent and specific results
                specific_search_response = None
                try:
                    # Try to get recent, specific results first
                    specific_search_response = exa_client.search(
                        query=query,
                        num_results=self.max_results,
                        use_autoprompt=True,
                        include_domains=None,
                        exclude_domains=None,
                        start_published_date="2023-01-01", # Focus on recent results
                    )
                    print(f"Specific search response received: {type(specific_search_response)}")
                except Exception as specific_e:
                    print(f"Specific Exa search API error: {specific_e}")
                
                # Fallback to a more general search if needed
                general_search_response = None
                if specific_search_response is None or not hasattr(specific_search_response, 'results') or not specific_search_response.results:
                    try:
                        # Fall back to general search if specific search fails or returns no results
                        print("Falling back to general search")
                        general_search_response = exa_client.search(
                            query=query,
                            num_results=self.max_results,
                            use_autoprompt=True,
                            include_domains=None,
                            exclude_domains=None,
                        )
                        print(f"General search response received: {type(general_search_response)}")
                    except Exception as general_e:
                        print(f"General Exa search API error: {general_e}")
                        time.sleep(1)
                        continue
                
                # Use specific results if available, otherwise use general results
                search_response = specific_search_response if (specific_search_response and 
                                                              hasattr(specific_search_response, 'results') and 
                                                              specific_search_response.results) else general_search_response
                
                # Safety check for None response
                if search_response is None:
                    print("Received None response from both search attempts")
                    time.sleep(1)
                    continue
                
                # Debug the search response
                print(f"Response type: {type(search_response)}")
                print(f"Response has results attribute: {hasattr(search_response, 'results')}")
                if hasattr(search_response, 'results'):
                    results = search_response.results
                    print(f"Results type: {type(results)}")
                    print(f"Results length: {len(results) if results is not None else 'None'}")
                    
                    # Check if results is None or empty
                    if results is None or len(results) == 0:
                        print("Results is None or empty")
                        time.sleep(1)
                        continue
                    
                    # Check the first result for debugging
                    if len(results) > 0:
                        first_result = results[0]
                        print(f"First result type: {type(first_result)}")
                        print(f"First result has title: {hasattr(first_result, 'title') if first_result is not None else 'Result is None'}")
                
                # Convert search_response.results to a list of dictionaries
                if hasattr(search_response, 'results') and search_response.results:
                    # Process results either concurrently or sequentially based on configuration
                    if self.concurrent_processing:
                        results_list = self._process_results_concurrently(search_response.results)
                    else:
                        results_list = self._process_results_sequentially(search_response.results)
                    
                    # Return empty list if no valid results were found
                    if not results_list:
                        print("No valid results were extracted from the response")
                        time.sleep(1)
                        continue
                        
                    print(f"Found {len(results_list)} valid search results")
                    
                    # Final check to ensure we're returning a valid list
                    if not isinstance(results_list, list):
                        print(f"Warning: results_list is not a list but {type(results_list)}")
                        results_list = list(results_list) if hasattr(results_list, '__iter__') else []
                    
                    print(f"Search completed in {time.time() - start_time:.2f} seconds")
                    return results_list
                else:
                    # Check if results attribute exists but is empty
                    if hasattr(search_response, 'results'):
                        print("Search response has empty results list")
                    else:
                        print("Search response does not have 'results' attribute")
                    
                    # Debug what attributes are available
                    attrs = [attr for attr in dir(search_response) if not attr.startswith('_')]
                    print(f"Available attributes: {attrs}")
                    
                    # Wait before retrying
                    time.sleep(1)
                    continue
                    
            except Exception as e:
                print(f"Error in Exa search (attempt {attempt+1}/{self.max_retries}): {e}")
                time.sleep(1)
                
                # On the last attempt, just return empty results
                if attempt == self.max_retries - 1:
                    return []
        
        print("All search attempts failed, returning empty results")
        print(f"Search failed after {time.time() - start_time:.2f} seconds")
        return []  # Return empty list if all attempts failed
    
    def _process_results_concurrently(self, search_results) -> List[Dict[str, Any]]:
        """
        Process search results concurrently for improved performance.
        
        Args:
            search_results: The raw search results from Exa
            
        Returns:
            Processed list of result dictionaries
        """
        results_list = []
        with ThreadPoolExecutor(max_workers=min(10, len(search_results))) as executor:
            # Submit all processing tasks
            future_to_index = {
                executor.submit(self._process_single_result, i, result): i 
                for i, result in enumerate(search_results)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result_dict = future.result()
                    if result_dict:
                        results_list.append(result_dict)
                        print(f"Processed result {index+1}: {result_dict.get('title', '')[:50]}...")
                except Exception as e:
                    print(f"Error processing result at index {index}: {e}")
        
        return results_list
    
    def _process_results_sequentially(self, search_results) -> List[Dict[str, Any]]:
        """
        Process search results sequentially.
        
        Args:
            search_results: The raw search results from Exa
            
        Returns:
            Processed list of result dictionaries
        """
        results_list = []
        for i, result in enumerate(search_results):
            try:
                result_dict = self._process_single_result(i, result)
                if result_dict:
                    results_list.append(result_dict)
                    print(f"Processed result {i+1}: {result_dict.get('title', '')[:50]}...")
            except Exception as e:
                print(f"Error processing result at index {i}: {e}")
        
        return results_list
    
    def _process_single_result(self, index: int, result: Any) -> Dict[str, Any]:
        """
        Process a single search result safely.
        
        Args:
            index: The index of the result
            result: The search result from Exa
            
        Returns:
            Processed result dictionary or None if invalid
        """
        if result is None:
            print(f"Warning: Encountered None result at index {index}")
            return None
            
        try:
            result_dict = {}
            # Convert result object to dictionary with proper error handling
            result_dict['title'] = self.get_attribute(result, 'title', 'No title')
            result_dict['url'] = self.get_attribute(result, 'url', 'No URL')
            result_dict['published_date'] = self.get_attribute(result, 'published_date', 'Unknown date')
            result_dict['text'] = self.get_attribute(result, 'text', 'No content available')
            
            # No longer trying to extract highlights since they're not supported
            
            return result_dict
        except Exception as e:
            print(f"Error processing result at index {index}: {e}")
            return None
    
    def get_attribute(self, obj, attr, default):
        """
        Safely get attribute from an object or dictionary.
        
        Args:
            obj: The object or dictionary
            attr: The attribute or key name
            default: Default value if attribute doesn't exist
            
        Returns:
            The attribute value or default
        """
        if obj is None:
            return default
            
        if isinstance(obj, dict):
            return obj.get(attr, default)
        else:
            return getattr(obj, attr, default)
    
    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results for extraction.
        
        Args:
            results: List of search results from Exa
            
        Returns:
            Formatted search results as a string
        """
        if not results:
            return "No search results were found for this query."
            
        formatted_results = ""
        
        for i, result in enumerate(results, 1):
            try:
                if result is None:
                    print(f"Skipping None result at position {i}")
                    continue
                    
                formatted_results += f"Result {i}:\n"
                
                # Get title and URL safely
                title = self.get_attribute(result, 'title', 'No title')
                url = self.get_attribute(result, 'url', 'No URL')
                published_date = self.get_attribute(result, 'published_date', 'Unknown date')
                
                formatted_results += f"Title: {title}\n"
                formatted_results += f"URL: {url}\n"
                formatted_results += f"Published: {published_date}\n"
                
                # Handle text content with extra care
                text = self.get_attribute(result, 'text', None)
                if text is None:
                    formatted_results += "Content: No content available\n\n"
                else:
                    # Safely get a substring of text
                    try:
                        # Use longer text excerpts for better context (up to 1000 chars)
                        text_preview = text[:1000] + "..." if len(text) > 1000 else text
                        formatted_results += f"Content: {text_preview}\n\n"
                    except Exception as text_e:
                        print(f"Error extracting text preview: {text_e}")
                        formatted_results += "Content: Error extracting content\n\n"
                
                # Removed highlights section since highlights aren't supported
                
            except Exception as e:
                print(f"Error formatting result at position {i}: {e}")
                formatted_results += f"Result {i}: Error formatting this result\n\n"
        
        if not formatted_results:
            return "No search results were found for this query."
            
        return formatted_results
    
    def extract_information(self, query: str, search_results: str) -> str:
        """
        Extract comprehensive information from search results.
        
        Args:
            query: The original query
            search_results: Formatted search results
            
        Returns:
            Extracted and synthesized information
        """
        try:
            start_time = time.time()
            
            if search_results == "No search results were found for this query.":
                return f"No specific information was found regarding '{query}'. This could indicate that either no such information exists, or that more specialized sources might be needed for this query."
            
            print(f"Extracting comprehensive information for query: '{query}'")
            
            # Break the extraction into chunks if the search results are very large
            # to avoid potential token limits and improve extraction quality
            result_length = len(search_results)
            print(f"Total search results length: {result_length} characters")
            
            if result_length > 30000:
                print("Search results are very large, breaking into chunks for processing")
                # Process in multiple chunks and combine results
                results = self._extract_from_large_results(query, search_results)
            else:
                # Process normally for reasonable-sized results
                results = self.extraction_chain.invoke({
                    "query": query,
                    "search_results": search_results
                })
            
            print(f"Information extraction completed in {time.time() - start_time:.2f} seconds")
            return results
        except Exception as e:
            print(f"Error in information extraction: {e}")
            traceback_str = traceback.format_exc()
            print(f"Traceback: {traceback_str}")
            
            # Provide a more helpful error message
            if "token" in str(e).lower():
                return f"The search results were too extensive to process at once. Please try a more specific query to narrow down the information needed."
            else:
                return f"Failed to extract information: {str(e)}"
    
    def _extract_from_large_results(self, query: str, search_results: str) -> str:
        """
        Process large search results by breaking them into manageable chunks.
        
        Args:
            query: The original query
            search_results: The complete search results
            
        Returns:
            Combined extracted information
        """
        # Split the results by "Result X:" markers
        result_sections = []
        current_section = ""
        
        for line in search_results.split('\n'):
            if line.startswith("Result ") and ":" in line and len(current_section) > 0:
                result_sections.append(current_section)
                current_section = line + "\n"
            else:
                current_section += line + "\n"
                
        if current_section:
            result_sections.append(current_section)
            
        print(f"Split search results into {len(result_sections)} sections")
        
        # Process each batch of results separately
        batches = []
        batch_size = 3  # Process 3 results at a time
        for i in range(0, len(result_sections), batch_size):
            batch = result_sections[i:i+batch_size]
            batch_text = "\n".join(batch)
            batches.append(batch_text)
        
        print(f"Created {len(batches)} batches for processing")
        
        # Extract information from each batch
        extracted_sections = []
        for i, batch in enumerate(batches):
            print(f"Processing batch {i+1}/{len(batches)}")
            try:
                batch_results = self.extraction_chain.invoke({
                    "query": f"{query} (analyzing results {i*batch_size+1}-{min((i+1)*batch_size, len(result_sections))})",
                    "search_results": batch
                })
                extracted_sections.append(batch_results)
            except Exception as batch_e:
                print(f"Error processing batch {i+1}: {batch_e}")
                extracted_sections.append(f"Error processing this segment of results: {str(batch_e)}")
        
        # Combine the extracted information
        combined_results = self._combine_extracted_sections(query, extracted_sections)
        return combined_results
    
    def _combine_extracted_sections(self, query: str, extracted_sections: List[str]) -> str:
        """
        Combine multiple extracted sections into a unified response.
        
        Args:
            query: The original query
            extracted_sections: List of extracted information sections
            
        Returns:
            Combined and synthesized information
        """
        if not extracted_sections:
            return f"No information could be extracted regarding '{query}'."
            
        if len(extracted_sections) == 1:
            return extracted_sections[0]
            
        # For multiple sections, use another LLM call to synthesize them
        print("Combining multiple information sections into a unified response")
        
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research synthesizer. Your task is to combine multiple sections of research information into a single, coherent, comprehensive response.

Maintain all factual information from each section, but remove redundancies and organize the information logically.
Create appropriate sections with clear headings to structure the information.
Preserve all source citations in the format [Source Title: URL].
Focus on providing a complete answer to the original query.
"""),
            ("human", """
Original Query: {query}

Here are sections of information extracted from different batches of search results:

{sections}

Synthesize these sections into a single, comprehensive response that thoroughly answers the original query.
""")
        ])
        
        synthesis_chain = synthesis_prompt | model | StrOutputParser()
        
        # Join the sections with clear separators
        formatted_sections = ""
        for i, section in enumerate(extracted_sections, 1):
            formatted_sections += f"SECTION {i}:\n{section}\n\n{'='*50}\n\n"
        
        try:
            combined_result = synthesis_chain.invoke({
                "query": query,
                "sections": formatted_sections
            })
            return combined_result
        except Exception as synth_e:
            print(f"Error in synthesis: {synth_e}")
            # Fall back to simple concatenation if synthesis fails
            return "# Combined Research Results\n\n" + "\n\n## Next Section\n\n".join(extracted_sections)
    
    def research(self, query: str) -> Dict[str, Any]:
        """
        Perform comprehensive web research for the given query.
        
        Args:
            query: The user query
            
        Returns:
            Dictionary containing research results and metadata
        """
        total_start_time = time.time()
        try:
            # Perform search
            print(f"Starting search for query: {query}")
            search_results = self.search(query)
            print(f"Search returned {len(search_results) if search_results else 0} results")
            
            # Check search results type and format
            if search_results is None:
                print("Warning: search_results is None, converting to empty list")
                search_results = []
            
            if not isinstance(search_results, list):
                print(f"Warning: search_results is not a list but {type(search_results)}, converting to list")
                try:
                    search_results = list(search_results) if hasattr(search_results, '__iter__') else []
                except Exception as convert_e:
                    print(f"Error converting search_results to list: {convert_e}")
                    search_results = []
            
            # Format search results
            try:
                format_start = time.time()
                formatted_results = self.format_search_results(search_results)
                print(f"Formatted results length: {len(formatted_results) if formatted_results else 0}")
                print(f"Formatting completed in {time.time() - format_start:.2f} seconds")
            except Exception as format_e:
                print(f"Error formatting search results: {format_e}")
                formatted_results = "No search results were found for this query."
            
            # Extract information
            try:
                print("Extracting information from search results...")
                extracted_info = self.extract_information(query, formatted_results)
                print("Information extraction completed")
            except Exception as extract_e:
                print(f"Error extracting information: {extract_e}")
                extracted_info = f"Failed to extract information: {str(extract_e)}"
            
            # Collect sources with extra safety checks
            sources = []
            if search_results:
                try:
                    print("Processing sources...")
                    for i, result in enumerate(search_results):
                        if result is None:
                            print(f"Skipping None result at index {i}")
                            continue
                        
                        # Extra defensive coding - check if result is of expected type
                        if not isinstance(result, dict):
                            print(f"Result at index {i} is not a dictionary but {type(result)}")
                            
                            # If result is not a dict, try to convert it or create a placeholder
                            if hasattr(result, '__dict__'):
                                print(f"Converting object to dictionary at index {i}")
                                result = result.__dict__
                            else:
                                print(f"Creating placeholder for result at index {i}")
                                result = {}
                        
                        title = self.get_attribute(result, "title", "No title")
                        url = self.get_attribute(result, "url", "No URL")
                        published_date = self.get_attribute(result, "published_date", "Unknown date")
                        
                        source = {
                            "title": title, 
                            "url": url,
                            "published_date": published_date
                        }
                        sources.append(source)
                    
                    print(f"Processed {len(sources)} sources")
                except Exception as source_e:
                    print(f"Error processing sources: {source_e}")
                    # Continue with empty sources rather than failing
                    sources = []
            
            # Determine success based on whether we got useful information and it's not just an error message
            success = len(search_results) > 0 and not extracted_info.startswith("Failed to extract information")
            error_message = "" if success else "No search results found or extraction failed"
            
            result = {
                "query": query,
                "success": success,
                "error": error_message,
                "information": extracted_info,
                "sources": sources
            }
            
            print(f"Research completed successfully in {time.time() - total_start_time:.2f} seconds")
            return result
        except Exception as e:
            total_time = time.time() - total_start_time
            print(f"Error in web research: {e}")
            traceback_str = traceback.format_exc()
            print(f"Traceback: {traceback_str}")
            print(f"Research failed after {total_time:.2f} seconds")
            return {
                "query": query,
                "success": False,
                "error": str(e),
                "information": f"An error occurred while researching '{query}': {str(e)}. Please try a different query or approach.",
                "sources": []
            }

# Test the web research agent
if __name__ == "__main__":
    agent = WebResearchAgent(max_results=8, concurrent_processing=True)
    start = time.time()
    result = agent.research("What are the latest developments in quantum computing?")
    end = time.time()
    
    print(f"\nQuery: {result['query']}")
    print(f"Success: {result['success']}")
    print(f"Total research time: {end - start:.2f} seconds")
    
    if result['success']:
        print("\nInformation:")
        print(result['information'])
        
        print("\nSources:")
        for source in result['sources']:
            print(f"- {source['title']}: {source['url']}")
    else:
        print(f"Error: {result['error']}")
        if result.get('information'):
            print(f"Information: {result['information']}")
