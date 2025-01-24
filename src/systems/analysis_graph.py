from typing import Dict, Any, List, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain.prompts import PromptTemplate
import logging
import os
from src.systems.types import SystemState
from src.utils.timing import measure_time
import json
from openai import OpenAI

from .vector_store import VectorStore

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

class TrendAnalysisAgent(BaseAnalysisAgent):
    @property
    def agent_id(self) -> str:
        return "trend"

    @property
    def capability(self) -> AgentCapability:
        return AgentCapability(
            id=self.agent_id,
            keywords=["trend", "change", "over time", "pattern", "historical"],
            description="Analyzes trends and patterns in data over time",
            priority=1,
        )

    def analyze(self, query: str, retrieval_chain: RetrievalQAWithSourcesChain) -> str:
        prompt = """You are a trend analysis agent specializing in insurance data analysis.
        Your role is to identify and explain significant trends in insurance costs and expenditures over time.
        Focus on:
        1. Year-over-year changes
        2. Long-term patterns
        3. Significant turning points
        4. Rate of change analysis
        
        Below you will find insurance cost data. Analyze this data to answer the question.
        Provide specific numbers and percentages to support your analysis.
        Always cite the years you're referring to.
        
        {context}
        
        Question: {question}
        
        Answer: Let me analyze this based on the data provided."""
        
        # Create enhanced chain with the prompt
        enhanced_chain = self._enhance_retrieval_chain(retrieval_chain, prompt)
        
        # Run analysis with document retrieval
        logger.info(f"Running trend analysis for query: {query}")
        try:
            result = enhanced_chain.invoke({
                "question": query
            })
            logger.info(f"Trend analysis result: {result}")
            return f"Trend Analysis:\n\n{result['answer']}\n\nSources: {result['sources']}"
        except Exception as e:
            logger.error(f"Error in trend analysis: {str(e)}")
            raise

class ComparisonAgent(BaseAnalysisAgent):
    @property
    def agent_id(self) -> str:
        return "comparison"

    @property
    def capability(self) -> AgentCapability:
        return AgentCapability(
            id=self.agent_id,
            keywords=["compare", "difference", "versus", "vs", "between"],
            description="Compares metrics between different periods or categories",
            priority=2,
        )

    def analyze(self, query: str, retrieval_chain: RetrievalQAWithSourcesChain) -> str:
        prompt = """You are a comparison agent specializing in insurance data analysis.
        Your role is to compare insurance metrics between different time periods.
        Focus on:
        1. Direct numerical comparisons
        2. Percentage differences
        3. Relative changes
        
        Below you will find insurance cost data. Compare this data to answer the question.
        Always show your calculations and cite the specific years being compared.
        
        {context}
        
        Question: {question}
        
        Answer: Let me analyze this based on the data provided."""
        
        # Create enhanced chain with the prompt
        enhanced_chain = self._enhance_retrieval_chain(retrieval_chain, prompt)
        
        # Run analysis with document retrieval
        logger.info(f"Running comparison analysis for query: {query}")
        try:
            result = enhanced_chain.invoke({
                "question": query
            })
            logger.info(f"Comparison analysis result: {result}")
            return f"Comparison Analysis:\n\n{result['answer']}\n\nSources: {result['sources']}"
        except Exception as e:
            logger.error(f"Error in comparison analysis: {str(e)}")
            raise

class SummaryAgent(BaseAnalysisAgent):
    @property
    def agent_id(self) -> str:
        return "summary"

    @property
    def capability(self) -> AgentCapability:
        return AgentCapability(
            id=self.agent_id,
            keywords=["summarize", "overview", "brief", "summary", "key points"],
            description="Provides high-level summaries of the data",
            priority=0,
        )

    def analyze(self, query: str, retrieval_chain: RetrievalQAWithSourcesChain) -> str:
        prompt = """You are a summary agent specializing in insurance data analysis.
        Your role is to provide clear, concise summaries of insurance data.
        Focus on:
        1. Key statistics and figures
        2. Overall patterns
        3. Important highlights
        4. Notable outliers
        
        Below you will find insurance cost data. Summarize this data to answer the question.
        Keep the summary focused and data-driven.
        
        {context}
        
        Question: {question}
        
        Answer: Let me analyze this based on the data provided."""
        
        # Create enhanced chain with the prompt
        enhanced_chain = self._enhance_retrieval_chain(retrieval_chain, prompt)
        
        # Run analysis with document retrieval
        logger.info(f"Running summary analysis for query: {query}")
        try:
            result = enhanced_chain.invoke({
                "question": query
            })
            logger.info(f"Summary analysis result: {result}")
            return f"Summary:\n\n{result['answer']}\n\nSources: {result['sources']}"
        except Exception as e:
            logger.error(f"Error in summary analysis: {str(e)}")
            raise

class AnalysisAgent(Runnable):
    """Main analysis agent that coordinates other specialized agents"""
    NODE_NAME = "analysis"

    def __init__(self, vector_store: VectorStore):
        """Initialize the analysis agent with necessary components"""
        self.vector_store = vector_store
        
        # Initialize LLMs
        self.agents_selection_llm = OpenAI()

        self.advanced_llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.0
        )

        self.basic_llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0
        )

        # Initialize retrieval chain
        prompt_template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        {context}
        
        Question: {question}
        Answer: """

        self.retrieval_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.advanced_llm,
            chain_type="stuff",
            retriever=self.vector_store.store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 6,
                }
            ),
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                ),
                "document_variable_name": "context",
            },
            return_source_documents=True,
            verbose=True
        )

        # Register available agents
        self.agents: List[BaseAnalysisAgent] = [
            TrendAnalysisAgent(self.advanced_llm),
            ComparisonAgent(self.advanced_llm),
            SummaryAgent(self.basic_llm)
        ]

    def _select_agents_with_llm(self, query: str) -> List[BaseAnalysisAgent]:
        """Use LLM to select appropriate agents based on query content"""
        # Prepare agent descriptions for LLM
        agent_descriptions = "\n".join([
            f"- {agent.agent_id}: {agent.capability.description}"
            for agent in self.agents
        ])

        try:
            # Get LLM's agent selection with structured output
            response = self.agents_selection_llm.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that selects appropriate analysis agents. You must respond with a JSON object containing 'selected_agents' (array of agent IDs) and 'reasoning' (string)."
                    },
                    {
                        "role": "user", 
                        "content": f"""Given a user's query about insurance data analysis, determine which analysis agents would be most appropriate.

Available agents:
{agent_descriptions}

User query: "{query}"

Analyze the query and select up to {MAX_SELECTED_AGENTS} most relevant agents. Consider:
1. The type of analysis requested
2. The specific information needs
3. The complexity of the query"""
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "agent_selection_response",
                        "description": "Select appropriate agents for analyzing data based on the query",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "description": "Response from the agent selection LLM",
                            "properties": {
                                "selected_agents": {"type": "array", "items": {"type": "string"}},
                                "reasoning": {
                                    "type": "string", 
                                    "description": "Brief explanation of why these agents were selected"
                                    },
                            },
                            "required": ["selected_agents", "reasoning"],
                            "additionalProperties": False
                        }
                    }
                }
            )

            # Parse the JSON response
            result = json.loads(response.choices[0].message.content)
            selected_agent_ids = result["selected_agents"]
            logger.info(f"Agent selection reasoning: {result['reasoning']}")
            
            # Map selected IDs to actual agents
            selected_agents = []
            for agent_id in selected_agent_ids:
                matching_agents = [agent for agent in self.agents if agent.agent_id == agent_id]
                if matching_agents:
                    selected_agents.append(matching_agents[0])

            # Fallback to SummaryAgent if no agents were selected
            if not selected_agents:
                selected_agents = [next(agent for agent in self.agents if isinstance(agent, SummaryAgent))]

            return selected_agents
            
        except Exception as e:
            logger.error(f"Error in agent selection: {e}")
            # Fallback to SummaryAgent on error
            return [next(agent for agent in self.agents if isinstance(agent, SummaryAgent))]

    @measure_time
    def invoke(
        self,
        state: SystemState,
        config: RunnableConfig | None = None,
    ) -> SystemState:
        """Process state in the workflow"""
        try:
            # Extract query from state
            query = next(msg.content for msg in reversed(state.messages) if isinstance(msg, HumanMessage))
            logger.info(f"Processing query: {query}")

            # Select appropriate agents using LLM
            selected_agents = self._select_agents_with_llm(query)
            logger.info(f"Selected agents: {[agent.__class__.__name__ for agent in selected_agents]}")
            
            # Run analysis with each selected agent
            for agent in selected_agents:
                logger.info(f"Running analysis with {agent.__class__.__name__}")
                result = agent.analyze(query, self.retrieval_chain)
                logger.info(f"Analysis result from {agent.__class__.__name__}:\n{result}")
                state.messages.append(AIMessage(content=result))
                logger.info(f"Current state messages count: {len(state.messages)}")

            return state

        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            state.error = str(e)
            return state
