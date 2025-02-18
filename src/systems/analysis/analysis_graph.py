from typing import List
from dataclasses import dataclass
from abc import ABC, abstractmethod
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain.prompts import PromptTemplate
import logging
import os
from src.systems.types import SystemState, WorkflowNode
from src.systems.vector_store import VectorStore
from src.systems.types import Status
from src.systems.analysis.multi_agent_supervisor import MultiAgentSupervisor

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

    def __init__(self, llm: BaseChatModel):
        """Initialize the agent with an LLM"""
        self.llm = llm

    @abstractmethod
    def analyze(self, query: str, retrieval_chain: RetrievalQAWithSourcesChain) -> str:
        """Run analysis on the query"""
        pass

    def _enhance_retrieval_chain(self, retrieval_chain: RetrievalQAWithSourcesChain, prompt: str) -> RetrievalQAWithSourcesChain:
        """Enhance retrieval chain with agent-specific prompt"""
        # Create a new chain with the updated prompt
        enhanced_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,  # Use the LLM passed to the constructor
            chain_type="stuff",
            retriever=retrieval_chain.retriever,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=prompt,
                    input_variables=["context", "question"]
                ),
                "document_variable_name": "context",
            },
            return_source_documents=True,
            verbose=True
        )
        return enhanced_chain

class AnalysisAgent(Runnable, WorkflowNode):
    """Main analysis agent that coordinates other specialized agents"""
    
    @property
    def NODE_NAME(self) -> str:
        return "analysis_agent"
        
    @property
    def STEP_DESCRIPTION(self) -> str:
        return "Data Analysis"

    def __init__(self, vector_store: VectorStore):
        """Initialize the analysis agent with necessary components"""
        self.vector_store = vector_store
        
        # Disable logging for HTTP requests
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("openai").setLevel(logging.ERROR)
        logging.getLogger("anthropic").setLevel(logging.ERROR)
        logging.getLogger("httpcore").setLevel(logging.ERROR)
        
        # Initialize supervisor
        self.supervisor = MultiAgentSupervisor(vector_store)

    def invoke(
        self,
        state: SystemState,
        config: RunnableConfig | None = None,
    ) -> SystemState:
        """Process state in the workflow"""
        self.start_processing()
        try:
            # Extract query from state
            query = next(msg.content for msg in reversed(state.messages) if isinstance(msg, HumanMessage))
            if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug(f"Processing query: {query}")

            # Run analysis with supervisor
            results = self.supervisor.analyze(query)
            
            # Add results to messages
            for result in results:
                state.messages.append(result)
                logger.info(result.content)
                
            if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug(f"Added {len(results)} analysis results to state")

            state.status = Status.ANALYSIS_COMPLETE
            self._stop_timing()
            return state
            
        except Exception as e:
            logger.error(f"Error in analysis agent: {str(e)}")
            state.set_error(str(e))
            return state
