from typing import Dict, Any, List, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod
from langchain_core.language_models import BaseChatModel
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
import logging
from src.systems.vector_store import VectorStore
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain

logger = logging.getLogger(__name__)

@dataclass
class AgentCapability:
    """Describes what an agent can do"""
    id: str           # Unique identifier for the agent
    keywords: List[str]  # Keywords that indicate this agent should be used
    description: str     # Description of what this agent does
    priority: int       # Priority when multiple agents match (higher = more priority)

class BaseAnalysisAgent(ABC):
    """Base class for all analysis agents"""
    
    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Unique identifier for the agent"""
        pass

    @property
    @abstractmethod
    def capability(self) -> AgentCapability:
        """Define agent's capabilities"""
        pass

    @property
    @abstractmethod
    def prompt_template(self) -> str:
        """Define agent's prompt template"""
        pass

    def __init__(self, llm: BaseChatModel):
        """Initialize the agent with an LLM"""
        self.llm = llm
        self.chain = self._create_chain()
    
    @abstractmethod
    def _create_chain(self) -> RunnableSequence:
        """Create the analysis chain for this agent"""
        pass
        
    def _prepare_context(self, query: str, vector_store: VectorStore) -> str:
        """Retrieve and format relevant context for analysis"""
        try:
            if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug(f"Retrieving documents for query: {query}")
            
            # Get relevant documents
            docs = vector_store.store.similarity_search(
                query,
                k=6,
                fetch_k=10  # Fetch more candidates for better relevance
            )
            
            if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug(f"Retrieved {len(docs)} documents")
                if docs:
                    logger.debug(f"First document metadata: {docs[0].metadata}")
            
            if not docs:
                logger.warning("No documents found in similarity search")
                return "No relevant context found."
            
            # Format context from documents
            context_parts = []
            for i, doc in enumerate(docs):
                if logger.getEffectiveLevel() <= logging.DEBUG:
                    logger.debug(f"Processing document {i+1}/{len(docs)}")
                
                if "structured_data" in doc.metadata:
                    # Format structured data
                    data = doc.metadata["structured_data"]
                    for entry in data:
                        if all(k in entry for k in ["year", "expenditure", "percent_change"]):
                            context_parts.append(
                                f"Year: {entry['year']}\n"
                                f"Expenditure: ${entry['expenditure']:,.2f}\n"
                                f"Change: {entry['percent_change']}%\n"
                            )
                else:
                    # Use raw content if no structured data
                    context_parts.append(doc.page_content)
            
            context = "\n\n".join(context_parts)
            
            if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug(f"Formatted context length: {len(context)}")
                logger.debug(f"Number of data points: {len(context_parts)}")
            
            return context
            
        except Exception as e:
            logger.error(f"Error preparing context: {str(e)}")
            return f"Error retrieving context: {str(e)}"

    def analyze(self, query: str, vector_store: VectorStore) -> str:
        """Run analysis on the query using retrieved context"""
        try:
            if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug(f"Creating retrieval chain for {self.agent_id}")
            
            # Get context first
            context = self._prepare_context(query, vector_store)
            
            if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug(f"Context prepared, length: {len(context) if context else 0}")
                logger.debug(f"First 200 chars of context: {context[:200] if context else 'None'}")
            
            # Create retrieval chain with agent's own LLM
            retrieval_chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=self.llm,  # Use agent's specific LLM
                chain_type="stuff",
                retriever=vector_store.store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 6}
                ),
                chain_type_kwargs={
                    "prompt": PromptTemplate(
                        template=self.prompt_template,
                        input_variables=["context", "question"]
                    ),
                    "document_variable_name": "context",
                    "verbose": False
                },
                return_source_documents=True,
                verbose=False
            )
            
            if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug("Running analysis chain...")
            
            # Run chain with both query and prepared context
            result = retrieval_chain.invoke({
                "question": query,
                "context": context
            })
            
            if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug(f"Chain completed. Result type: {type(result)}")
                logger.debug(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
            
            # Format result with agent type
            return f"{self.agent_id.title()} Analysis:\n\n{result['answer']}"
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            return f"Analysis error: {str(e)}" 