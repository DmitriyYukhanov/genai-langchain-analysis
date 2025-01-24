from typing import List, Optional
from pydantic import BaseModel
from langchain_core.documents import Document
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from abc import ABC, abstractmethod

class WorkflowNode(ABC):
    """Base class for all workflow nodes"""
    
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

class SystemState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    role: str
    input_path: Optional[str] = None
    processed_documents: Optional[List[Document]] = None
    vectors_stored: bool = False
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True  # To allow Document type