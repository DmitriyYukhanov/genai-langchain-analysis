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

class MetadataExtractor:
    def __init__(self):
        pass

    def extract(self, content: str) -> Dict[str, Any]:
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
            lines = self._clean_lines(content)
            i = 0
            while i < len(lines):
                line = lines[i]
                if self._is_header_line(line):
                    i += 1
                    continue

                if self._is_year_line(line):
                    if self._is_valid_multiline(lines, i):
                        year, expenditure, percent_change = self._extract_multiline(lines, i)
                        self._update_metadata(metadata, year, expenditure, percent_change)
                        i += 3
                        continue

                if self._is_valid_singleline(line):
                    year, expenditure, percent_change = self._extract_singleline(line)
                    self._update_metadata(metadata, year, expenditure, percent_change)

                i += 1

            self._finalize_metadata(metadata)
        except Exception as e:
            logger.debug(f"Error extracting metadata: {str(e)}")
        
        return metadata

    def _clean_lines(self, content: str) -> List[str]:
        return [line.strip() for line in content.split('\n') if line.strip()]

    def _is_header_line(self, line: str) -> bool:
        return any(header in line.lower() for header in ['year', 'average expenditure', 'percent change'])

    def _is_year_line(self, line: str) -> bool:
        return line.isdigit() and 1900 < int(line) < 2100

    def _is_valid_multiline(self, lines: List[str], index: int) -> bool:
        if index + 2 >= len(lines):
            return False
        exp_line = lines[index + 1]
        change_line = lines[index + 2]
        return self._is_expenditure_line(exp_line) and self._is_percent_change_line(change_line)

    def _is_expenditure_line(self, line: str) -> bool:
        return (line.startswith('$') or 
                line.replace('.', '').replace(',', '').isdigit() or
                line.replace('.', '').replace(',', '').replace('-', '').isdigit())

    def _is_percent_change_line(self, line: str) -> bool:
        return (line.endswith('%') or 
                line.replace('.', '').replace('-', '').isdigit())

    def _extract_multiline(self, lines: List[str], index: int) -> tuple[int, float, float]:
        year = int(lines[index])
        expenditure = float(lines[index + 1].replace('$', '').replace(',', ''))
        percent_change = float(lines[index + 2].replace('%', ''))
        return year, expenditure, percent_change

    def _is_valid_singleline(self, line: str) -> bool:
        parts = line.split()
        if len(parts) < 3:
            return False
        return (self._is_year_line(parts[0]) and 
                self._is_expenditure_line(parts[1]) and 
                self._is_percent_change_line(parts[2]))

    def _extract_singleline(self, line: str) -> tuple[int, float, float]:
        parts = line.split()
        year = int(parts[0])
        expenditure = float(parts[1].replace('$', '').replace(',', ''))
        percent_change = float(parts[2].replace('%', ''))
        return year, expenditure, percent_change

    def _update_metadata(self, metadata: Dict[str, Any], year: int, expenditure: float, percent_change: float):
        metadata["years"].append(year)
        metadata["expenditures"][year] = expenditure
        metadata["percent_changes"][year] = percent_change

    def _finalize_metadata(self, metadata: Dict[str, Any]):
        if metadata["years"]:
            metadata["years"] = sorted(list(set(metadata["years"])))
            metadata["year_range"] = f"{min(metadata['years'])}-{max(metadata['years'])}"
            metadata["average_expenditure"] = sum(metadata["expenditures"].values()) / len(metadata["expenditures"])
            metadata["total_percent_change"] = sum(metadata["percent_changes"].values())

class DocumentProcessor(Runnable):
    NODE_NAME:str = "doc_processor"

    def __init__(self):
        """Initialize the document processor with text splitter configuration"""
        super().__init__()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        self.metadata_extractor = MetadataExtractor()

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
        return self.metadata_extractor.extract(content)

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
                state.error = "No input path specified"
                return state

            path = Path(state.input_path)
            
            if path.is_file():
                documents = self.load_excel(str(path))
                processed_docs = self.process_documents(documents)
            elif path.is_dir():
                processed_docs = self.process_directory(str(path))
            else:
                state.error = f"Invalid input path: {state.input_path}"
                return state

            if processed_docs:
                state.processed_documents = processed_docs
                return state
            
            state.error = "No documents were processed"
            return state
                
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}")
            state.error = str(e)
            state.processed_documents = None
            state.vectors_stored = False
            return state

