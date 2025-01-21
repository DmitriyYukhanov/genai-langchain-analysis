import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
import logging
from langchain_core.runnables import Runnable, RunnableConfig
from sqlalchemy import create_engine
from src.types import AgentState
from src.utils.timing import measure_time

logger = logging.getLogger(__name__)

class VectorStore(Runnable):
    NODE_NAME = "vector_store"

    def __init__(self):
        """Initialize the vector store with OpenAI embeddings and PGVector"""
        self.connection_string = os.getenv("PGVECTOR_CONNECTION_STRING")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.connection_string:
            raise ValueError("PGVECTOR_CONNECTION_STRING environment variable is required")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        try:
            # Using OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=self.openai_api_key
            )
            self.collection_name = "insurance_docs"

            self.store = PGVector(
                    embeddings=self.embeddings,
                    collection_name=self.collection_name,
                    connection=self.connection_string,
                    use_jsonb=True,
                )
            logger.info(f"Successfully initialized vector store: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to vector store"""
        logger.info(f"Adding {len(documents)} documents to vector store")
        if not hasattr(self, 'store'):
            self.init_store()
        try:
            self.store.add_documents(documents)
            logger.info("Successfully added documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search"""
        logger.info(f"Performing similarity search for: {query}")
        if not hasattr(self, 'store'):
            self.init_store()
        try:
            results = self.store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents")
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
