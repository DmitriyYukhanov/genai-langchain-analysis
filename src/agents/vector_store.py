import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
import logging
from langchain_core.runnables import Runnable, RunnableConfig
from sqlalchemy import create_engine
from src.types import AgentState
from src.utils.timing import measure_time
import time

logger = logging.getLogger(__name__)

class VectorStore(Runnable):
    NODE_NAME = "vector_store"
    SEARCH_TIMEOUT = 2.0  # Maximum search time in seconds

    def __init__(self):
        """Initialize the vector store with OpenAI embeddings and PGVector"""
        self.database_connection_string = os.getenv("PGVECTOR_CONNECTION_STRING")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.database_connection_string:
            raise ValueError("PGVECTOR_CONNECTION_STRING environment variable is required")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        try:
            # Using OpenAIEmbeddings with caching for efficiency
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=self.openai_api_key,
                max_retries=2  # Limit retries for faster failure
            )
            self.collection_name = "insurance_docs"

            # Initialize PGVector with optimized settings
            self.store = PGVector(
                embeddings=self.embeddings,
                collection_name=self.collection_name,
                connection=self.database_connection_string,
                use_jsonb=True,  # Use JSONB for better metadata handling
                pre_delete_collection=True,  # Ensure clean state
                distance_strategy="cosine"  # Use cosine similarity for better matching
            )
            logger.info(f"Successfully initialized vector store: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def _prepare_metadata(self, documents: List[Document]) -> List[Document]:
        """Prepare and validate document metadata before storage"""
        for doc in documents:
            # Ensure basic metadata fields exist
            if 'source' not in doc.metadata:
                doc.metadata['source'] = 'unknown'
            if 'created_at' not in doc.metadata:
                doc.metadata['created_at'] = time.time()
            
            # Add processing metadata
            doc.metadata['chunk_length'] = len(doc.page_content)
            doc.metadata['processed_at'] = time.time()
            
            # Ensure all metadata values are JSON serializable
            for key, value in doc.metadata.items():
                if isinstance(value, (set, tuple)):
                    doc.metadata[key] = list(value)
        return documents

    @measure_time
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to vector store with metadata validation"""
        logger.info(f"Adding {len(documents)} documents to vector store")
        try:
            # Prepare and validate metadata
            processed_docs = self._prepare_metadata(documents)
            
            # Batch documents for efficient processing
            batch_size = 100
            for i in range(0, len(processed_docs), batch_size):
                batch = processed_docs[i:i + batch_size]
                self.store.add_documents(batch)
                logger.info(f"Added batch of {len(batch)} documents")
            
            logger.info("Successfully added all documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    @measure_time
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search with timeout"""
        logger.info(f"Performing similarity search for: {query}")
        try:
            start_time = time.time()
            
            # Perform search with timeout check
            results = self.store.similarity_search(
                query,
                k=k,
                filter=None
            )
            
            # Check if search time exceeds limit
            search_time = time.time() - start_time
            if search_time > self.SEARCH_TIMEOUT:
                logger.warning(f"Search exceeded time limit: {search_time:.2f}s > {self.SEARCH_TIMEOUT}s")
            
            logger.info(f"Found {len(results)} similar documents in {search_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            return []

    @measure_time
    def invoke(
        self,
        state: AgentState,
        config: RunnableConfig | None = None,
    ) -> AgentState:
        """Process state in the workflow"""
        try:
            if state.processed_documents:
                self.add_documents(state.processed_documents)
                state.vectors_stored = True
                return state
            
            state.error = "No processed documents found"
            return state
        
        except Exception as e:
            logger.error(f"Error in vector store: {str(e)}")
            state.error = str(e)
            return state
