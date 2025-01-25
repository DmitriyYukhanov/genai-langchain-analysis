import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
import logging
from langchain_core.runnables import Runnable, RunnableConfig
from sqlalchemy import create_engine
from src.systems.types import SystemState, WorkflowNode
from src.utils.progress import ProgressManager, parallel_process
import time
from src.systems.types import Status

logger = logging.getLogger(__name__)

class VectorStore(Runnable, WorkflowNode):
    """Stores and retrieves document vectors using PGVector"""
    
    @property
    def NODE_NAME(self) -> str:
        return "vector_store"
        
    @property
    def STEP_DESCRIPTION(self) -> str:
        return "Vector Storage"

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
                max_retries=2,  # Limit retries for faster failure
                show_progress_bar=False  # Disable progress bar
            )
            self.collection_name = "insurance_docs"

            # Initialize PGVector with optimized settings
            self.store = PGVector(
                embeddings=self.embeddings,
                collection_name=self.collection_name,
                connection=self.database_connection_string,
                use_jsonb=True,  # Use JSONB for better metadata handling
                pre_delete_collection=False,
                distance_strategy="cosine"  # Use cosine similarity for better matching
            )
            logger.debug(f"Initialized vector store: {self.collection_name}")
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

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to vector store with metadata validation and progress tracking"""
        logger.debug(f"Adding {len(documents)} documents to vector store")
        try:
            # Prepare and validate metadata
            processed_docs = self._prepare_metadata(documents)
            
            # Process in batches for memory efficiency
            batch_size = 100
            total_batches = (len(processed_docs) + batch_size - 1) // batch_size
            
            with ProgressManager() as progress:
                store_task = progress.add_task(
                    description=f"Storing {len(processed_docs)} document vectors",
                    total_steps=len(processed_docs)
                )
                
                for i in range(0, len(processed_docs), batch_size):
                    batch = processed_docs[i:i + batch_size]
                    if logger.getEffectiveLevel() <= logging.DEBUG:
                        logger.debug(f"Adding batch of {len(batch)} documents to PGVector")
                    
                    # Add documents without verification
                    self.store.add_documents(batch)
                    logger.info(f"Added batch of {len(batch)} documents")
                    
                    # Update progress
                    progress.update_task(
                        store_task,
                        completed=float(min(i + batch_size, len(processed_docs))),
                        description=f"Storing document vectors - batch {(i//batch_size)+1}/{total_batches}"
                    )
                
                # Ensure 100% completion
                progress.update_task(
                    store_task,
                    completed=float(len(processed_docs)),
                    description="Vector storage complete"
                )
                progress.remove_task(store_task)
            
        except Exception as e:
            logger.error(f"Error storing document vectors: {str(e)}")
            raise

    def invoke(
        self,
        state: SystemState,
        config: RunnableConfig | None = None,
    ) -> SystemState:
        """Process state in the workflow"""
        self.start_processing()
        try:
            if state.processed_documents:
                self.add_documents(state.processed_documents)
                state.vectors_stored = True
                state.status = Status.VECTORS_SAVED
                self._stop_timing()
                return state
            
            state.set_error("No processed documents found")
            return state
        
        except Exception as e:
            logger.error(f"Error in vector store: {str(e)}")
            state.set_error(str(e))
            return state
