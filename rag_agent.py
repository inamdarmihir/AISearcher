import os
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Load environment variables
load_dotenv()

# Initialize the OpenAI model and embeddings
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)
embeddings = OpenAIEmbeddings()

class RAGAgent:
    """
    Agent for Retrieval-Augmented Generation using Qdrant vector database.
    """
    
    def __init__(self, collection_name: str = "KnowledgeBase", top_k: int = 5):
        """
        Initialize the RAG agent.
        
        Args:
            collection_name: Name of the Qdrant collection
            top_k: Number of documents to retrieve
        """
        self.collection_name = collection_name
        self.top_k = top_k
        
        # Initialize Qdrant client
        self.client = self._init_qdrant_client()
        
        # Create the collection if it doesn't exist
        self._create_collection_if_not_exists()
        
        # Create the synthesis prompt
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledge base agent that synthesizes information from retrieved documents.
            
Your task is to analyze the provided documents and extract the most relevant information related to the user's query.
Focus on providing a comprehensive and accurate answer based solely on the retrieved information.

Organize the information in a coherent manner, and cite the sources for each piece of information.
"""),
            ("human", """
Query: {query}

Retrieved Documents:
{documents}

Synthesize the information to provide a comprehensive answer to the query.
""")
        ])
        
        # Create the synthesis chain
        self.synthesis_chain = self.synthesis_prompt | model | StrOutputParser()
    
    def _init_qdrant_client(self) -> QdrantClient:
        """
        Initialize the Qdrant client.
        
        Returns:
            Qdrant client
        """
        try:
            # Check if Qdrant URL and API key are provided
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            
            if qdrant_url:
                # Connect to remote Qdrant instance
                if qdrant_api_key:
                    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                else:
                    client = QdrantClient(url=qdrant_url)
            else:
                # Use local Qdrant instance or in-memory storage
                client = QdrantClient(":memory:")
            
            return client
        except Exception as e:
            print(f"Error initializing Qdrant client: {e}")
            raise
    
    def _create_collection_if_not_exists(self) -> None:
        """
        Create the Qdrant collection if it doesn't exist.
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                # Create the collection
                # Get dimension size from OpenAI embeddings
                vector_size = 1536  # OpenAI embeddings dimension
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created collection: {self.collection_name}")
            else:
                print(f"Collection {self.collection_name} already exists")
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the Qdrant collection.
        
        Args:
            documents: List of documents to add
        """
        try:
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            
            points = []
            
            # Process each document
            for doc in documents:
                chunks = text_splitter.split_text(doc.page_content)
                
                # Get embeddings for chunks
                embeddings_list = embeddings.embed_documents(chunks)
                
                # Add each chunk to Qdrant
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_list)):
                    # Create a unique ID for each chunk
                    doc_id = str(uuid.uuid4())
                    
                    # Prepare metadata
                    metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
                    metadata['chunk_id'] = i
                    
                    # Create point for batch upload
                    points.append(
                        PointStruct(
                            id=doc_id,
                            vector=embedding,
                            payload={
                            "content": chunk,
                            "metadata": str(metadata),
                            "source": metadata.get("source", "unknown")
                            }
                        )
                    )
            
            # Upload points in batch
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                    )
            
            print(f"Added {len(documents)} documents to collection {self.collection_name}")
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        Query the knowledge base for relevant documents.
        
        Args:
            query: The user query
            
        Returns:
            Dictionary containing query results and metadata
        """
        try:
            # Generate query embedding
            query_embedding = embeddings.embed_query(query)
            
            # Query Qdrant
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=self.top_k
            )
            
            # Extract documents
            documents = []
            for scored_point in search_result:
                content = scored_point.payload.get("content", "")
                metadata_str = scored_point.payload.get("metadata", "{}")
                source = scored_point.payload.get("source", "unknown")
                
                documents.append(f"Document (Source: {source}):\n{content}\n")
            
            if not documents:
                return {
                    "query": query,
                    "success": False,
                    "error": "No relevant documents found",
                    "answer": None,
                    "documents": []
                }
            
            # Format documents for synthesis
            formatted_documents = "\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents)])
            
            # Synthesize information
            answer = self.synthesis_chain.invoke({
                "query": query,
                "documents": formatted_documents
            })
            
            return {
                "query": query,
                "success": True,
                "answer": answer,
                "documents": documents
            }
        except Exception as e:
            print(f"Error querying knowledge base: {e}")
            return {
                "query": query,
                "success": False,
                "error": str(e),
                "answer": None,
                "documents": []
            }
    
    def add_sample_data(self) -> None:
        """
        Add sample data to the knowledge base for demonstration purposes.
        """
        sample_documents = [
            Document(
                page_content="Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations. The devices that perform quantum computations are known as quantum computers.",
                metadata={"source": "quantum_computing_intro.txt"}
            ),
            Document(
                page_content="Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.",
                metadata={"source": "ai_definition.txt"}
            ),
            Document(
                page_content="Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil, and gas, which produces heat-trapping gases.",
                metadata={"source": "climate_change_overview.txt"}
            ),
            Document(
                page_content="The theory of relativity usually encompasses two interrelated theories by Albert Einstein: special relativity and general relativity. Special relativity applies to all physical phenomena in the absence of gravity. General relativity explains the law of gravitation and its relation to other forces of nature.",
                metadata={"source": "theory_of_relativity.txt"}
            ),
            Document(
                page_content="Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.",
                metadata={"source": "machine_learning_basics.txt"}
            )
        ]
        
        self.add_documents(sample_documents)
        print("Added sample data to the knowledge base.")

# Test the RAG agent
if __name__ == "__main__":
    agent = RAGAgent()
    agent.add_sample_data()
    result = agent.query("Explain quantum computing in simple terms")
    
    print(f"Query: {result['query']}")
    print(f"Success: {result['success']}")
    
    if result['success']:
        print("\nAnswer:")
        print(result['answer'])
        
        print("\nRetrieved Documents:")
        for i, doc in enumerate(result['documents'], 1):
            print(f"Document {i}:")
            print(doc)
    else:
        print(f"Error: {result['error']}")
