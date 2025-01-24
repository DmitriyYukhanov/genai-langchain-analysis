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
from src.systems.analysis.factory import AgentFactory
from src.systems.vector_store import VectorStore
from src.systems.types import Status

logger = logging.getLogger(__name__)

# Configuration constants
MAX_SELECTED_AGENTS = 2

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
        
        # Initialize LLMs with logging control
        self.advanced_llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.0,
            verbose=False  # Always keep quiet
        )

        self.basic_llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0,
            verbose=False,  # Always keep quiet
            request_timeout=30  # Only OpenAI needs timeout
        )

        # Initialize agent factory
        self.agent_factory = AgentFactory(self.advanced_llm, self.basic_llm)

    def _run_analysis(self, query: str) -> List[str]:
        """Run analysis using selected agents"""
        try:
            # Select appropriate agents using factory
            if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug("Selecting appropriate agents for analysis...")
            selected_agents = self.agent_factory.select_agents(query)
            
            if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug(f"Selected agents: {[agent.__class__.__name__ for agent in selected_agents]}")
            
            # Run analysis with each selected agent
            results = []
            for agent in selected_agents:
                if logger.getEffectiveLevel() <= logging.DEBUG:
                    logger.debug(f"Running analysis with {agent.__class__.__name__}")
                
                # Run agent analysis with vector store
                result = agent.analyze(query, self.vector_store)
                
                if logger.getEffectiveLevel() <= logging.DEBUG:
                    logger.debug(f"Completed {agent.__class__.__name__} analysis")
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            return [f"Analysis error: {str(e)}"]

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

            # Run analysis with selected agents
            results = self._run_analysis(query)
            
            # Add results to messages
            for result in results:
                # Format result to be more readable
                formatted_result = f"\nAnalysis Result:\n{result}"
                state.messages.append(AIMessage(content=formatted_result))
                # Log result in non-debug mode
                logger.info(formatted_result)
                
            if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug(f"Added {len(results)} analysis results to state")

            state.status = Status.ANALYSIS_COMPLETE
            self._stop_timing()
            return state
            
        except Exception as e:
            logger.error(f"Error in analysis agent: {str(e)}")
            state.set_error(str(e))
            return state
