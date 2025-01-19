from typing import List, Dict, Any
from pathlib import Path
from langchain_community.document_loaders import UnstructuredExcelLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableConfig
import logging
import glob
import os
from src.types import AgentState
from src.utils.file_detection import detect_file_type

logger = logging.getLogger(__name__)

class DocumentProcessor(Runnable):
    def __init__(self, vector_store):
        """Initialize the document processor with text splitter configuration"""
        super().__init__()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        self.vector_store = vector_store

    def load_excel(self, file_path: str) -> List[Document]:
        """
        Load Excel or HTML file using appropriate loader
        
        Args:
            file_path: Path to the Excel/HTML file
            
        Returns:
            List of Document objects
        """
        logger.info(f"Loading file: {file_path}")
        try:
            file_type = detect_file_type(file_path)

            logger.info(f"File type detected: {file_type}")
            
            if file_type == 'html':
                loader = BSHTMLLoader(file_path)
            else:
                loader = UnstructuredExcelLoader(
                    file_path,
                    mode="elements",
                    strategy="fast"
                )
                
            documents = loader.load()
            
            # Add source and basic metadata
            for doc in documents:
                doc.metadata.update({
                    "source": file_path,
                    "file_name": os.path.basename(file_path),
                    "file_type": file_type
                })
                
            logger.info(f"Successfully loaded {len(documents)} elements from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return []

    def extract_metadata(self, content: str) -> Dict[str, Any]:
        """
        Extract metadata from document content
        
        Args:
            content: Document content string
            
        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}
        
        # Extract years
        try:
            years = [int(word) for word in content.split() 
                    if word.isdigit() and 1900 < int(word) < 2100]
            if years:
                metadata["years"] = sorted(list(set(years)))
                metadata["year_range"] = f"{min(years)}-{max(years)}"
        except Exception as e:
            logger.debug(f"Could not extract years: {str(e)}")
            
        # Extract monetary values
        try:
            monetary = [word for word in content.split() 
                       if word.startswith('$') or word.replace('.', '').replace(',', '').isdigit()]
            if monetary:
                metadata["monetary_values"] = monetary
        except Exception as e:
            logger.debug(f"Could not extract monetary values: {str(e)}")
            
        return metadata

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents by splitting and extracting metadata
        
        Args:
            documents: List of raw documents
            
        Returns:
            List of processed Document objects
        """
        logger.info(f"Processing {len(documents)} documents")
        try:
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)
            
            # Extract and enhance metadata
            for i, doc in enumerate(split_docs):
                # Add chunk information
                doc.metadata["chunk_id"] = i
                
                # Extract additional metadata
                metadata = self.extract_metadata(doc.page_content)
                doc.metadata.update(metadata)
                
            logger.info(f"Successfully processed documents into {len(split_docs)} chunks")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return documents

    def process_directory(self, directory_path: str) -> List[Document]:
        """
        Process all Excel/HTML files in a directory
        
        Args:
            directory_path: Path to directory containing files
            
        Returns:
            List of processed Document objects
        """
        logger.info(f"Processing directory: {directory_path}")
        all_documents = []
        
        # Find all Excel and HTML files
        patterns = ['*.xls*', '*.html', '*.htm']
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(directory_path, pattern)))
        
        for file_path in files:
            try:
                documents = self.load_excel(file_path)
                if documents:
                    processed_docs = self.process_documents(documents)
                    all_documents.extend(processed_docs)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
                
        return all_documents

    def invoke(
        self,
        state: AgentState,
        config: RunnableConfig | None = None,
    ) -> AgentState:
        """
        Process documents and update state
        """
        try:
            if not state.input_path:
                state.next = "end"
                state.error = "No input path specified"
                return state

            path = Path(state.input_path)
            
            if path.is_file():
                documents = self.load_excel(str(path))
                processed_docs = self.process_documents(documents)
            elif path.is_dir():
                processed_docs = self.process_directory(str(path))
            else:
                state.next = "end"
                state.error = f"Invalid input path: {state.input_path}"
                return state

            if processed_docs:
                state.next = "vector_store"
                state.processed_documents = processed_docs
                return state
            
            state.next = "end"
            state.error = "No documents were processed"
            return state
                
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}")
            return AgentState(
                content=state.content,
                role=state.role,
                next="end",
                input_path=state.input_path,
                processed_documents=None,
                vectors_stored=False,
                error=str(e)
            )
