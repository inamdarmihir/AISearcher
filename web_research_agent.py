import os
import time
import traceback
import re
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
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

# Pydantic model for structured research output
class ResearchOutput(BaseModel):
    """Structured format for research information extracted from sources."""
    summary: str = Field(..., description="A brief 1-2 sentence summary of the findings")
    main_findings: List[str] = Field(..., description="List of 3-5 key findings related to the query")
    detailed_analysis: str = Field(..., description="Comprehensive analysis of the information found")
    sources: List[Dict[str, str]] = Field(..., description="List of sources used, each with title and URL")

# PII patterns for identification
PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone_number': r'\b(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'credit_card': r'\b(?:\d{4}[- ]?){3}\d{4}\b',
    'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
}

# Input guardrails - blocked topics and patterns
BLOCKED_TOPICS = [
    'hack', 'exploit', 'vulnerability', 'illegal', 'bomb', 'terrorist',
    'child abuse', 'pornography', 'murder', 'suicide', 'self-harm',
]

# Privacy-sensitive query patterns - detect requests for personal information
PRIVACY_PATTERNS = [
    # Critical privacy concerns - always block
    (r'\b(home|house|residential|private|personal|family)\s+(address|location|residence)\b', 'home address', 'critical'),
    (r'\b(where|location).+(live|lives|living|reside|resides|residing)\b', 'residence location', 'critical'),
    (r'\b(address|location).+(celebrity|actor|actress|athlete|player|politician|star)\b', 'celebrity address', 'critical'),
    (r'\b(phone|contact|email|social)\s+(number|details|address|information)\b', 'contact information', 'critical'),
    (r'\b(credit\s+card|bank\s+account|financial)\s+(details|information|data)\b', 'financial information', 'critical'),
    (r'\b(SSN|social\s+security|national\s+id|passport|driver\'s\s+license)\b', 'identification numbers', 'critical'),
    
    # Moderate privacy concerns - configurable blocking
    (r'\b(date|day)\s+of\s+(birth|birthday)\b', 'date of birth', 'moderate'),
    (r'\b(birth|born).+(date|day|year)\b', 'birth date', 'moderate'),
    (r'\b(age|how\s+old)\b', 'age', 'moderate'),
    (r'\b(marital|marriage|relationship)\s+(status|history)\b', 'marital status', 'moderate'),
    (r'\b(salary|income|earnings|net\s+worth)\b', 'financial status', 'moderate'),
    (r'\b(family|children|kids|spouse|wife|husband|partner)\b', 'family details', 'moderate'),
    
    # Public information about public figures - warning only
    (r'\b(career|achievements|records|statistics|stats|biography|profile)\b', 'public profile', 'low'),
]

class WebResearchAgent:
    """
    Agent for performing web research using Exa API with guardrails and PII protection.
    """
    
    def __init__(self, max_results: int = 8, max_retries: int = 3, concurrent_processing: bool = True, 
                 enable_pii_detection: bool = True, enable_guardrails: bool = True, 
                 structured_output: bool = True, privacy_level: str = "high"):
        """
        Initialize the web research agent.
        
        Args:
            max_results: Maximum number of search results to retrieve
            max_retries: Maximum number of retries for API calls
            concurrent_processing: Whether to process results concurrently
            enable_pii_detection: Whether to enable PII detection and redaction
            enable_guardrails: Whether to enable input and output guardrails
            structured_output: Whether to use structured output format
            privacy_level: Level of privacy enforcement ('low', 'medium', 'high')
        """
        self.max_results = max_results
        self.max_retries = max_retries
        self.concurrent_processing = concurrent_processing
        self.enable_pii_detection = enable_pii_detection
        self.enable_guardrails = enable_guardrails
        self.structured_output = structured_output
        self.privacy_level = privacy_level.lower()
        
        # Validate privacy level
        if self.privacy_level not in ['low', 'medium', 'high']:
            print(f"Invalid privacy level '{privacy_level}', defaulting to 'high'")
            self.privacy_level = 'high'
        
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

IMPORTANT GUARDRAILS:
1. DO NOT include any personally identifiable information (PII) such as non-public email addresses, phone numbers, 
   physical addresses, or other private contact information.
2. DO NOT generate harmful, illegal, unethical or deceptive content.
3. If asked about topics like hacking, exploits, or illegal activities, refuse to provide specific details 
   that could enable harm.
4. Maintain a factual, neutral tone and avoid politically biased language.
5. Cite sources for all significant claims and avoid making unsubstantiated assertions.

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

        # Define structured output parser if enabled
        if self.structured_output:
            # Define the structured output prompt
            self.structured_extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert web research agent that extracts and organizes information from search results.
                
Your task is to analyze the provided search results and extract ALL relevant information related to the user's query.
Structure your response using the following format:
1. A brief summary (1-2 sentences)
2. 3-5 key findings as bullet points
3. A comprehensive analysis with clearly marked sections
4. A list of sources used

Follow the same research guidelines and guardrails as the standard extraction but format your output 
in a structured JSON format that follows the specified schema.

IMPORTANT GUARDRAILS:
1. DO NOT include any personally identifiable information (PII) such as email addresses, phone numbers, or addresses.
2. DO NOT generate harmful, illegal, unethical, or deceptive content.
3. Maintain a factual, neutral tone and avoid biased language.
4. Cite sources for all claims and avoid making unsubstantiated assertions.
"""),
                ("human", """
Query: {query}

Search Results:
{search_results}

Provide a comprehensive analysis in the required structured format.
""")
            ])
            
            # Create the extraction chain with structured output
            self.structured_extraction_chain = self.structured_extraction_prompt | model | JsonOutputParser()
        
        # Create the standard extraction chain
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
    
    def check_guardrails(self, query: str) -> Dict[str, Any]:
        """
        Check if a query violates input guardrails.
        
        Args:
            query: The query to check
            
        Returns:
            Dictionary with pass status and reason for failure if any
        """
        if not self.enable_guardrails:
            return {"pass": True, "reason": None}
            
        # Check for blocked topics
        query_lower = query.lower()
        for topic in BLOCKED_TOPICS:
            if topic in query_lower:
                return {
                    "pass": False,
                    "reason": f"Query contains potentially harmful or inappropriate content: '{topic}'",
                    "category": "harmful_content"
                }
        
        # Check for privacy-sensitive patterns with respect to configured privacy level
        for pattern, description, severity in PRIVACY_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                # Determine if this should be blocked based on privacy level
                should_block = False
                
                if severity == 'critical':
                    # Critical privacy concerns are always blocked
                    should_block = True
                elif severity == 'moderate':
                    # Moderate concerns blocked at medium and high privacy levels
                    should_block = self.privacy_level in ['medium', 'high']
                elif severity == 'low':
                    # Low concerns only blocked at high privacy level
                    should_block = self.privacy_level == 'high'
                
                if should_block:
                    return {
                        "pass": False,
                        "reason": f"Query appears to be requesting personal or private information ({description}). " +
                                "For privacy and ethical reasons, we cannot provide specific personal details about individuals.",
                        "category": "privacy_violation",
                        "severity": severity
                    }
                else:
                    # If not blocking, we'll still log the concern
                    print(f"Privacy concern detected but allowed by current privacy level: {description} (severity: {severity})")
        
        # Check query length - prevent excessively long queries
        if len(query) > 1000:
            return {
                "pass": False,
                "reason": "Query exceeds maximum allowed length (1000 characters)",
                "category": "excessive_length"
            }
            
        # Check for excessive special characters or potential code injection
        special_char_percentage = sum(1 for c in query if not c.isalnum() and not c.isspace()) / len(query) if query else 0
        if special_char_percentage > 0.3:  # If more than 30% are special characters
            return {
                "pass": False,
                "reason": "Query contains an unusually high percentage of special characters",
                "category": "suspicious_characters"
            }
            
        return {"pass": True, "reason": None, "category": None}
    
    def detect_and_redact_pii(self, text: str) -> str:
        """
        Detect and redact personally identifiable information (PII) from text.
        
        Args:
            text: The text to process
            
        Returns:
            Text with PII redacted
        """
        if not self.enable_pii_detection or not text:
            return text
            
        redacted_text = text
        
        # Apply each PII detection pattern
        for pii_type, pattern in PII_PATTERNS.items():
            # Find all matches
            matches = re.finditer(pattern, redacted_text)
            
            # Replace matches with redacted indicator
            for match in matches:
                pii_value = match.group()
                redacted_text = redacted_text.replace(pii_value, f"[REDACTED {pii_type.upper()}]")
                
        return redacted_text
    
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
            
            # Add a secondary privacy check for patterns that weren't blocked
            # but should get a disclaimer - useful for 'allowed but concerning' queries
            privacy_disclaimer = ""
            
            # Only check lower severity patterns that we didn't block earlier
            for pattern, description, severity in [p for p in PRIVACY_PATTERNS if p[2] in ['low', 'moderate']]:
                if re.search(pattern, query.lower(), re.IGNORECASE):
                    # For public figures, we may still allow the query but add a disclaimer
                    if self._is_likely_public_figure(query):
                        privacy_disclaimer = f"""
> **Note on Personal Information**: This information about {self._extract_name(query)} is publicly available 
> as they are a public figure. We only provide information that is already in the public domain.
> We respect privacy and do not share private personal details that are not publicly disclosed.
"""
                        break
            
            print(f"Extracting comprehensive information for query: '{query}'")
            
            # Break the extraction into chunks if the search results are very large
            # to avoid potential token limits and improve extraction quality
            result_length = len(search_results)
            print(f"Total search results length: {result_length} characters")
            
            # Process differently based on size and output format
            if result_length > 30000:
                print("Search results are very large, breaking into chunks for processing")
                # Process in multiple chunks and combine results
                results = self._extract_from_large_results(query, search_results)
            else:
                # Process based on output format preference
                if self.structured_output:
                    try:
                        print("Using structured output format")
                        results_dict = self.structured_extraction_chain.invoke({
                            "query": query,
                            "search_results": search_results
                        })
                        
                        # Convert structured output to formatted string
                        results = self._format_structured_output(results_dict)
                    except Exception as struct_e:
                        print(f"Error in structured extraction: {struct_e}, falling back to standard extraction")
                        # Fall back to standard extraction on error
                        results = self.extraction_chain.invoke({
                            "query": query,
                            "search_results": search_results
                        })
                else:
                    # Standard extraction
                    results = self.extraction_chain.invoke({
                        "query": query,
                        "search_results": search_results
                    })
            
            # Apply PII detection if enabled
            if self.enable_pii_detection:
                results = self.detect_and_redact_pii(results)
            
            # Add privacy disclaimer if present
            if privacy_disclaimer:
                # Insert after the first heading if possible
                if "# " in results:
                    parts = results.split("# ", 1)
                    results = parts[0] + "# " + parts[1].split("\n", 1)[0] + "\n" + privacy_disclaimer + "\n" + parts[1].split("\n", 1)[1]
                else:
                    results = privacy_disclaimer + "\n" + results
            
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
    
    def _format_structured_output(self, results_dict: Dict[str, Any]) -> str:
        """
        Format structured output dictionary into a readable string format.
        
        Args:
            results_dict: Dictionary containing structured research results
            
        Returns:
            Formatted string version of the structured results
        """
        try:
            # Handle the case where we might get a string instead of a dict
            if isinstance(results_dict, str):
                try:
                    results_dict = json.loads(results_dict)
                except:
                    return results_dict  # Return as is if can't parse
            
            # Build the formatted output
            output = "# Research Results\n\n"
            
            # Add summary
            if "summary" in results_dict:
                output += f"## Summary\n{results_dict['summary']}\n\n"
            
            # Add key findings
            if "main_findings" in results_dict and results_dict["main_findings"]:
                output += "## Key Findings\n"
                for i, finding in enumerate(results_dict["main_findings"], 1):
                    output += f"{i}. {finding}\n"
                output += "\n"
            
            # Add detailed analysis
            if "detailed_analysis" in results_dict:
                output += f"## Detailed Analysis\n{results_dict['detailed_analysis']}\n\n"
            
            # Add sources
            if "sources" in results_dict and results_dict["sources"]:
                output += "## Sources\n"
                for source in results_dict["sources"]:
                    if isinstance(source, dict) and "title" in source and "url" in source:
                        output += f"- [{source['title']}]({source['url']})\n"
                    elif isinstance(source, str):
                        output += f"- {source}\n"
                output += "\n"
            
            return output
        except Exception as format_e:
            print(f"Error formatting structured output: {format_e}")
            # If formatting fails, try to return the raw dictionary as a string
            try:
                return json.dumps(results_dict, indent=2)
            except:
                return str(results_dict)
    
    def research(self, query: str) -> Dict[str, Any]:
        """
        Perform comprehensive web research for the given query.
        
        Args:
            query: The user query
            
        Returns:
            Dictionary containing research results and metadata
        """
        total_start_time = time.time()
        
        # Check input guardrails first if enabled
        if self.enable_guardrails:
            guardrail_check = self.check_guardrails(query)
            if not guardrail_check["pass"]:
                print(f"Guardrail violation detected: {guardrail_check['reason']}")
                privacy_message = ""
                
                # Generate a more specific message for privacy violations
                if guardrail_check.get("category") == "privacy_violation":
                    privacy_message = f"""# Privacy Protection Notice

This query appears to be requesting personal or private information ({guardrail_check.get('severity', 'high')} sensitivity). For privacy, ethical, and security reasons, we cannot process requests seeking personal details about individuals, including:

- Home addresses or residential locations
- Personal contact information (email addresses, phone numbers)
- Financial details or identification numbers
- Private personal data such as date of birth, age, marital status
- Other non-public personal information

Instead, we suggest:

1. Focusing on publicly available information about the individual's professional work, career, or achievements
2. Using official contact forms on websites if you need to reach someone professionally
3. Connecting through professional networking platforms where individuals choose to share contact information

We're committed to respecting privacy and adhering to responsible AI practices.
"""
                
                return {
                    "query": query,
                    "success": False,
                    "error": "Guardrail violation",
                    "information": privacy_message or f"Unable to process this query: {guardrail_check['reason']}. Please try a different query.",
                    "sources": [],
                    "is_structured": False
                }
        
        try:
            # Add retry mechanism for the entire research process
            for research_attempt in range(2):  # Try up to 2 times for the whole research process
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
                    
                    # If no results found and this is first attempt, try again with a reformulated query
                    if not search_results and research_attempt == 0:
                        print("No results found on first attempt, trying with alternative query formulation")
                        # Remove apostrophes and reformat
                        alternative_query = query.replace("'s", "").replace("'", "")
                        if alternative_query != query:
                            print(f"Retrying with alternative query: '{alternative_query}'")
                            search_results = self.search(alternative_query)
                            print(f"Alternative search returned {len(search_results) if search_results else 0} results")
                    
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
                    
                    # If this attempt was successful, return the results
                    if success:
                        result = {
                            "query": query,
                            "success": success,
                            "error": "",
                            "information": extracted_info,
                            "sources": sources,
                            "is_structured": self.structured_output
                        }
                        
                        print(f"Research completed successfully in {time.time() - total_start_time:.2f} seconds")
                        return result
                    
                    # If we failed but have more attempts, try again
                    if research_attempt < 1:
                        print(f"Research attempt {research_attempt+1} failed, retrying...")
                        time.sleep(1)  # Brief pause before retrying
                    else:
                        # Return the best we've got on the last attempt
                        error_message = "No search results found or extraction failed"
                        result = {
                            "query": query,
                            "success": False,
                            "error": error_message,
                            "information": extracted_info if not extracted_info.startswith("Failed") else f"No specific information was found regarding '{query}'. This could indicate that either no such information exists, or that more specialized sources might be needed for this query.",
                            "sources": sources,
                            "is_structured": self.structured_output
                        }
                        
                        print(f"Research completed with partial success in {time.time() - total_start_time:.2f} seconds")
                        return result
                        
                except Exception as attempt_e:
                    print(f"Error in research attempt {research_attempt+1}: {attempt_e}")
                    if research_attempt < 1:
                        print("Retrying the research process...")
                        time.sleep(1)
                    else:
                        raise  # Re-raise on the last attempt
            
            # This should not be reached due to the returns above
            return {
                "query": query,
                "success": False,
                "error": "Research process failed after all attempts",
                "information": f"Unable to find reliable information about '{query}' after multiple attempts. Please try rephrasing your query or try a different topic.",
                "sources": [],
                "is_structured": False
            }
            
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
                "sources": [],
                "is_structured": False
            }
    
    def _is_likely_public_figure(self, query: str) -> bool:
        """
        Determine if a query is likely about a public figure.
        
        Args:
            query: The query string
            
        Returns:
            Boolean indicating if the query likely refers to a public figure
        """
        # Extract the potential name from the query
        name = self._extract_name(query)
        if not name:
            return False
            
        # Check if we have search results for this name
        # (This is a simplistic approach - in production, you might check against a database of known public figures)
        return True  # For now, we'll assume names in queries are public figures
        
    def _extract_name(self, query: str) -> str:
        """
        Extract a potential name from the query.
        
        Args:
            query: The query string
            
        Returns:
            Extracted name or empty string
        """
        # Simple extraction - get the first part before possessive or preposition
        for splitter in ["'s", " of ", " for ", " about "]:
            if splitter in query:
                return query.split(splitter)[0].strip()
                
        # If no splitter found, try to extract the first 2-3 words as a name
        words = query.split()
        if len(words) >= 2:
            return " ".join(words[:min(3, len(words))])
            
        return query  # Just return the whole query if we can't extract a name

# Test the web research agent
if __name__ == "__main__":
    agent = WebResearchAgent(
        max_results=8, 
        concurrent_processing=True,
        enable_pii_detection=True,
        enable_guardrails=True,
        structured_output=True,
        privacy_level="high"  # Options: "low", "medium", "high"
    )
    start = time.time()
    result = agent.research("What are the latest developments in quantum computing?")
    end = time.time()
    
    print(f"\nQuery: {result['query']}")
    print(f"Success: {result['success']}")
    print(f"Structured output: {result.get('is_structured', False)}")
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
