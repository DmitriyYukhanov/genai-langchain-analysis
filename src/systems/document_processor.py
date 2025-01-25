from typing import List, Dict, Any
from pathlib import Path
from langchain_community.document_loaders import UnstructuredExcelLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableConfig
import logging
import glob
import os
from src.systems.types import SystemState, WorkflowNode
from src.utils.file_detection import detect_file_type
from src.utils.progress import ProgressManager, parallel_process
import time
from src.systems.types import Status

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

class DocumentProcessor(Runnable, WorkflowNode):
    """Processes documents by loading, splitting and extracting metadata"""
    
    @property
    def NODE_NAME(self) -> str:
        return "doc_processor"
        
    @property
    def STEP_DESCRIPTION(self) -> str:
        return "Document Processing"

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
        try:
            file_type = detect_file_type(file_path)
            logger.debug(f"Loading {file_type} file: {os.path.basename(file_path)}")
            
            if file_type == 'html':
                loader = BSHTMLLoader(file_path)
            else:
                loader = UnstructuredExcelLoader(
                    file_path,
                    mode="elements",
                    strategy="fast"
                )
                
            documents = loader.load()
            
            if not documents:
                logger.warning(f"No content found in file: {os.path.basename(file_path)}")
                return []
                
            # Add source and basic metadata
            for doc in documents:
                doc.metadata.update({
                    "source": file_path,
                    "file_name": os.path.basename(file_path),
                    "file_type": file_type
                })
                
            logger.debug(f"Loaded {len(documents)} elements from {os.path.basename(file_path)}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading file {os.path.basename(file_path)}: {str(e)}")
            return []

    def extract_metadata(self, content: str) -> Dict[str, Any]:
        return self.metadata_extractor.extract(content)

    def process_document(self, doc: Document) -> Document:
        """Process a single document by extracting metadata"""
        try:
            # Extract and enhance metadata
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
            return doc
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return doc

    def process_directory(self, directory_path: str) -> List[Document]:
        """
        Process all Excel/HTML files in a directory using batch processing
        to minimize memory usage.
        
        Args:
            directory_path: Path to directory containing files
            
        Returns:
            List of processed Document objects
        """
        # Find all Excel and HTML files
        patterns = ['*.xls*', '*.html', '*.htm']
        files = []
        for pattern in patterns:
            found_files = glob.glob(os.path.join(directory_path, pattern))
            logger.debug(f"Found {len(found_files)} files matching pattern {pattern}")
            files.extend(found_files)
        
        if not files:
            logger.error(f"No Excel or HTML files found in {directory_path}")
            return []
            
        logger.info(f"Processing {len(files)} files")
        
        # Process files in batches to control memory usage
        batch_size = 5  # Process 5 files at a time
        all_processed_docs = []
        successful_files = 0
        
        with ProgressManager() as progress:
            main_task = progress.add_task(
                description=f"Processing {len(files)} files",
                total_steps=len(files)
            )
            
            for i in range(0, len(files), batch_size):
                batch_files = files[i:i + batch_size]
                batch_successful = 0
                
                # Process batch in parallel
                batch_docs = parallel_process(
                    batch_files,
                    self.load_excel,
                    description=None,  # No progress bar for individual batches
                    max_workers=min(batch_size, os.cpu_count())
                )
                
                # Combine and process documents from this batch
                batch_combined = []
                for docs in batch_docs:
                    if docs:
                        batch_combined.extend(docs)
                        batch_successful += 1
                
                if batch_combined:
                    # Process the combined documents from this batch
                    processed_batch = self.process_documents(batch_combined)
                    all_processed_docs.extend(processed_batch)
                
                # Update progress and counts
                successful_files += batch_successful
                current_progress = min((i + len(batch_files)), len(files))
                progress.update_task(
                    main_task,
                    completed=float(current_progress),
                    description=f"Processing files - {successful_files}/{len(files)} successful"
                )
            
            # Ensure progress bar shows completion before removal
            progress.update_task(
                main_task,
                completed=float(len(files)),
                description=f"Completed - {successful_files}/{len(files)} files processed"
            )
            # Small delay to ensure completion is visible
            time.sleep(0.1)
            progress.remove_task(main_task)
        
        if not all_processed_docs:
            logger.error("No documents were successfully processed")
            return []
            
        logger.info(f"Successfully processed {len(all_processed_docs)} chunks from {successful_files} files")
        return all_processed_docs

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents by splitting and extracting metadata.
        Uses batch processing for large document sets.
        
        Args:
            documents: List of raw documents
            
        Returns:
            List of processed Document objects
        """
        try:
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)
            
            # Process chunks in batches to control memory usage
            chunk_batch_size = 100  # Process 100 chunks at a time
            processed_docs = []
            
            # Process chunks without progress bar since it's too fast
            for i in range(0, len(split_docs), chunk_batch_size):
                batch = split_docs[i:i + chunk_batch_size]
                
                # Process batch in parallel
                processed_batch = parallel_process(
                    batch,
                    self.process_document,
                    description=None,  # No progress bar needed
                    max_workers=min(32, os.cpu_count() * 2)
                )
                
                # Add chunk IDs relative to the full set
                for j, doc in enumerate(processed_batch):
                    doc.metadata["chunk_id"] = i + j
                
                processed_docs.extend(processed_batch)
                
                # Force garbage collection after each batch if needed
                if len(processed_docs) > 1000:
                    import gc
                    gc.collect()
            
            return processed_docs
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return documents

    def invoke(
        self,
        state: SystemState,
        config: RunnableConfig | None = None,
    ) -> SystemState:
        """
        Process documents and update state
        """
        self.start_processing()
        total_files = 0
        successful_files = 0
        
        try:
            if not state.input_path:
                state.set_error("No input path specified")
                return state

            path = Path(state.input_path)
            
            if path.is_file():
                total_files = 1
                documents = self.load_excel(str(path))
                if documents:
                    successful_files = 1
                processed_docs = self.process_documents(documents) if documents else []
                
            elif path.is_dir():
                processed_docs = self.process_directory(str(path))
                # File counts are tracked inside process_directory
                if processed_docs:
                    successful_files = len(set(doc.metadata["source"] for doc in processed_docs))
                    total_files = len(glob.glob(os.path.join(str(path), "*.[xh][tl][ms]*")))
                
            else:
                state.set_error(f"Invalid input path: {state.input_path}")
                return state

            if processed_docs:
                state.processed_documents = processed_docs
                logger.info(f"Files: {successful_files}/{total_files} processed successfully")
                state.status = Status.DOCS_PROCESSED
                self._stop_timing()
                return state
            
            state.set_error("No documents were processed")
            return state
                
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}")
            state.set_error(str(e))
            state.processed_documents = None
            state.vectors_stored = False
            return state

