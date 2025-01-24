from typing import List, Optional
from pydantic import BaseModel
from langchain_core.documents import Document
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from abc import ABC, abstractmethod
import time
import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)

class Status(str, Enum):
    """Pipeline execution status"""
    INIT = "init"                    # Initial state
    ERROR = "error"                  # Error occurred
    DOCS_PROCESSING = "processing"   # Currently processing documents
    DOCS_PROCESSED = "processed"     # Documents processed successfully
    VECTORS_SAVING = "saving"        # Currently saving vectors
    VECTORS_SAVED = "saved"          # Vectors saved successfully
    ANALYSIS_RUNNING = "analyzing"   # Currently running analysis
    ANALYSIS_COMPLETE = "complete"   # Analysis complete
    DONE = "done"                   # All processing complete

class WorkflowNode(ABC):
    """Base class for all workflow nodes"""
    
    def __init__(self):
        self._start_time = None
        self._elapsed_time = None
    
    @property
    @abstractmethod
    def NODE_NAME(self) -> str:
        """Unique identifier for the node"""
        pass
        
    @property
    @abstractmethod
    def STEP_DESCRIPTION(self) -> str:
        """Human-readable description of what this node does"""
        pass
        
    @property
    def elapsed_time(self) -> float:
        """Time taken to complete the node's work in seconds"""
        return self._elapsed_time if self._elapsed_time is not None else 0.0
        
    def _start_timing(self) -> None:
        """Start timing the node's execution"""
        self._start_time = time.time()
        
    def _stop_timing(self) -> None:
        """Stop timing and calculate elapsed time"""
        if self._start_time is not None:
            self._elapsed_time = time.time() - self._start_time
            logger.info(f"\n{self.STEP_DESCRIPTION} completed in {self.elapsed_time:.1f}s\n")
        
    def start_processing(self) -> None:
        """Start node processing - reset state and start timing"""
        self._elapsed_time = None
        self._start_timing()

class SystemState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    role: str
    input_path: Optional[str] = None
    processed_documents: Optional[List[Document]] = None
    vectors_stored: bool = False
    error: Optional[str] = None
    status: Status = Status.INIT

    def set_error(self, error_msg: str) -> None:
        """Set error state with message"""
        self.error = error_msg
        self.status = Status.ERROR

    class Config:
        arbitrary_types_allowed = True  # To allow Document type