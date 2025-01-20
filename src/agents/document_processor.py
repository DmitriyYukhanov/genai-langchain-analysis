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
from src.utils.timing import measure_time

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
        metadata = {
            "years": [],
            "expenditures": {},
            "percent_changes": {}
        }
        
        try:
            # Split content into lines and clean them
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # Skip header lines
                if any(header in line.lower() for header in ['year', 'average expenditure', 'percent change']):
                    i += 1
                    continue

                # Try multi-line format first
                if line.isdigit() and 1900 < int(line) < 2100:
                    # Check if we have enough lines ahead and they look like expenditure and percent change
                    if i + 2 < len(lines):
                        exp_line = lines[i + 1]
                        change_line = lines[i + 2]
                        
                        # Check if next line looks like expenditure (starts with $ or is numeric)
                        is_expenditure = (exp_line.startswith('$') or 
                                        exp_line.replace('.', '').replace(',', '').isdigit() or
                                        exp_line.replace('.', '').replace(',', '').replace('-', '').isdigit())
                        
                        # Check if third line looks like percent change (ends with % or is numeric)
                        is_percent = (change_line.endswith('%') or 
                                    change_line.replace('.', '').replace('-', '').isdigit())
                        
                        if is_expenditure and is_percent:
                            try:
                                year = int(line)
                                expenditure = float(exp_line.replace('$', '').replace(',', ''))
                                percent_change = float(change_line.replace('%', ''))
                                
                                metadata["years"].append(year)
                                metadata["expenditures"][year] = expenditure
                                metadata["percent_changes"][year] = percent_change
                                
                                # Skip the processed lines
                                i += 3
                                continue
                            except ValueError:
                                pass
                
                # Try single-line format
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        # First part should be year
                        if parts[0].isdigit() and 1900 < int(parts[0]) < 2100:
                            year = int(parts[0])
                            
                            # Second part should be expenditure (starts with $ or is numeric)
                            exp_str = parts[1].replace('$', '').replace(',', '')
                            if exp_str.replace('.', '').replace('-', '').isdigit():
                                expenditure = float(exp_str)
                                
                                # Third part should be percent change (ends with % or is numeric)
                                change_str = parts[2].replace('%', '')
                                if change_str.replace('.', '').replace('-', '').isdigit():
                                    percent_change = float(change_str)
                                    
                                    metadata["years"].append(year)
                                    metadata["expenditures"][year] = expenditure
                                    metadata["percent_changes"][year] = percent_change
                    except (ValueError, IndexError):
                        pass
                
                i += 1
            
            # Sort years and create year range
            if metadata["years"]:
                metadata["years"] = sorted(list(set(metadata["years"])))
                metadata["year_range"] = f"{min(metadata['years'])}-{max(metadata['years'])}"
                
                # Add additional statistics
                metadata["average_expenditure"] = sum(metadata["expenditures"].values()) / len(metadata["expenditures"])
                metadata["total_percent_change"] = sum(metadata["percent_changes"].values())
                
        except Exception as e:
            logger.debug(f"Error extracting metadata: {str(e)}")
        
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
                
                # Add structured data for embeddings
                structured_data = []
                for year in metadata.get("years", []):
                    year_data = {
                        "year": year,
                        "expenditure": metadata["expenditures"].get(year),
                        "percent_change": metadata["percent_changes"].get(year)
                    }
                    structured_data.append(year_data)
                
                doc.metadata["structured_data"] = structured_data
                
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

    @measure_time
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
