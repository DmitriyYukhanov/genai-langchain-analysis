from typing import List, Type
from langchain_core.language_models import BaseChatModel
from .agents.abstract.base import BaseAnalysisAgent
from .agents.trend import TrendAnalysisAgent
from .agents.comparison import ComparisonAnalysisAgent
from .agents.summary import SummaryAgent
import logging
import json
from openai import OpenAI

logger = logging.getLogger(__name__)

class AgentFactory:
    """Factory for creating and managing analysis agents"""
    
    # Maximum number of agents to select for a query
    MAX_SELECTED_AGENTS = 2

    def __init__(self, advanced_llm: BaseChatModel, basic_llm: BaseChatModel):
        """Initialize the factory with LLMs"""
        self.advanced_llm = advanced_llm
        self.basic_llm = basic_llm
        self.agents_selection_llm = OpenAI()
        
        # Register available agent types
        self._agent_types: List[Type[BaseAnalysisAgent]] = [
            TrendAnalysisAgent,
            ComparisonAnalysisAgent,
            SummaryAgent
        ]
        
        # Initialize agent instances
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize agent instances with appropriate LLMs"""
        self.agents: List[BaseAnalysisAgent] = []
        for agent_type in self._agent_types:
            # Use advanced LLM for high-priority agents, basic LLM for others
            llm = self.advanced_llm if agent_type in [TrendAnalysisAgent, ComparisonAnalysisAgent] else self.basic_llm
            self.agents.append(agent_type(llm))

    def select_agents(self, query: str) -> List[BaseAnalysisAgent]:
        """Select appropriate agents for the given query using LLM"""
        try:
            # Prepare agent descriptions for LLM
            agent_descriptions = "\n".join([
                f"- {agent.agent_id}: {agent.capability.description}"
                for agent in self.agents
            ])

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

Analyze the query and select up to {self.MAX_SELECTED_AGENTS} most relevant agents. Consider:
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
            if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug(f"Agent selection reasoning: {result['reasoning']}")
            
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