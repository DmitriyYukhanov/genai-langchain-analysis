from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
import logging

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

    def analyze(self, query: str, retrieval_chain: RetrievalQAWithSourcesChain) -> str:
        """Run analysis on the query"""
        try:
            # Create enhanced chain with the prompt
            enhanced_chain = self._enhance_retrieval_chain(retrieval_chain)
            
            # Run analysis with document retrieval
            logger.info(f"Running {self.agent_id} analysis for query: {query}")
            result = enhanced_chain.invoke({
                "question": query
            })
            logger.info(f"{self.agent_id} analysis result: {result}")
            return f"{self.agent_id.title()} Analysis:\n\n{result['answer']}\n\nSources: {result['sources']}"
            
        except Exception as e:
            logger.error(f"Error in {self.agent_id} analysis: {str(e)}")
            raise

    def _enhance_retrieval_chain(self, retrieval_chain: RetrievalQAWithSourcesChain) -> RetrievalQAWithSourcesChain:
        """Enhance retrieval chain with agent-specific prompt"""
        return RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retrieval_chain.retriever,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=self.prompt_template,
                    input_variables=["context", "question"]
                ),
                "document_variable_name": "context",
            },
            return_source_documents=True,
            verbose=True
        ) 