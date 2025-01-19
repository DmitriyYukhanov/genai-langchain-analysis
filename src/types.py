from typing import List, Optional
from pydantic import BaseModel
from langchain_core.documents import Document

class AgentState(BaseModel):
    content: str
    role: str
    next: str
    input_path: Optional[str] = None
    processed_documents: Optional[List[Document]] = None
    vectors_stored: bool = False
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True  # To allow Document type