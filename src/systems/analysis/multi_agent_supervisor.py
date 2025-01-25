from typing import Dict, List, Literal, TypedDict
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.types import Command
import os
import logging
from pydantic import BaseModel
from src.systems.analysis.agents.trend import TrendAnalysisAgent
from src.systems.analysis.agents.comparison import ComparisonAnalysisAgent
from src.systems.analysis.agents.summary import SummaryAgent
from src.systems.vector_store import VectorStore
from enum import Enum

logger = logging.getLogger(__name__)

SUPERVISOR_NODE ="supervisor";
END_NODE_ALIAS ="FINISH";

class SupervisorNodeName(str, Enum):
    SUPERVISOR_NODE = SUPERVISOR_NODE

class NodeName(str, Enum):
    TREND_ANALYST = TrendAnalysisAgent.agent_id
    COMPARISON_ANALYST = ComparisonAnalysisAgent.agent_id
    SUMMARY_ANALYST = SummaryAgent.agent_id
    END_NODE = END

class MessagesState(BaseModel):
    """State that contains messages and next agent to run"""
    messages: List[BaseMessage]
    next: str | None = None

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: str
    decision_response: str | None = None

class MultiAgentSupervisor:
    """Supervisor that coordinates multiple analysis agents"""
    
    def __init__(self, vector_store: VectorStore):
        """Initialize supervisor with agents"""
        # Initialize LLM for supervisor
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.0,
        )

        self.basic_llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0,
            verbose=False,
            request_timeout=30
        )
        
        # Initialize agents
        self.trend_agent = TrendAnalysisAgent(self.llm)
        self.comparison_agent = ComparisonAnalysisAgent(self.llm)
        self.summary_agent = SummaryAgent(self.basic_llm)
        
        self.vector_store = vector_store
        
        # Define available agents
        self.members = [
            self.trend_agent.agent_id,
            self.comparison_agent.agent_id,
            self.summary_agent.agent_id
        ]
        
        # Define routing options
        self.options = self.members + []
        
        # Create agent descriptions for prompt
        self.agent_descriptions = "\n".join([
            f"- {self.trend_agent.agent_id}: {self.trend_agent.capability.description}",
            f"- {self.comparison_agent.agent_id}: {self.comparison_agent.capability.description}",
            f"- {self.summary_agent.agent_id}: {self.summary_agent.capability.description}"
        ])
        
        # Create system prompt
        self.system_prompt = (
            "You are a supervisor tasked with managing a conversation between the"
            f" following workers:\n\n{self.agent_descriptions}\n\n"
            "Given the following user request, respond with the worker to act next."
            " Each worker will perform a task and respond with their results."
            f" When you believe all necessary analysis is complete, conclude your decision_response and put {END_NODE_ALIAS} to the 'next' field."
            "\n\nConsider each agent's capabilities and the user's needs carefully."
            "\n\nYour decision_response should be extremely concise (max 2-3 sentences) and to the point."
            " Focus only on the most important findings or next steps."
            " Avoid repeating what agents have already said."
            " You can use multiple agents in sequence if needed."
        )
        
        # Create the graph
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the agent workflow graph"""
        # Create graph with state
        builder = StateGraph(MessagesState)
        
        # Add supervisor node
        builder.add_node(SUPERVISOR_NODE, self._supervisor_node)
        
        # Add agent nodes, no need to add edges back to supervisor since we're using goto command
        builder.add_node(self.trend_agent.agent_id, self._trend_node)
        builder.add_node(self.comparison_agent.agent_id, self._comparison_node)
        builder.add_node(self.summary_agent.agent_id, self._summary_node)
        
        # Set entry point
        builder.set_entry_point(SUPERVISOR_NODE)
        
        # Compile graph
        return builder.compile()
    
    def _supervisor_node(self, state: MessagesState) -> Command[NodeName]:
        """Supervisor node that decides which agent to run next"""
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        # Add conversation history
        for msg in state.messages:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        
        # Get next agent from LLM
        response = self.llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
    
        if goto == END_NODE_ALIAS:
            goto = END
            logger.info("Supervisor: Analysis complete.")
            if response.get("decision_response"):
                return Command(
                    goto=goto,
                    update={
                        "next": goto,
                        "messages": state.messages + [
                            AIMessage(content=response["decision_response"], additional_kwargs={"name": "supervisor"})
                        ]
                    }
                )
        else:
            logger.info(f"Supervisor: Selecting {goto} for next analysis.")
            if response.get("decision_response"):
                logger.info(f"Supervisor: {response['decision_response']}")
        
        return Command(goto=goto, update={"next": goto})
    
    def _trend_node(self, state: MessagesState) -> Command[SupervisorNodeName]:
        """Run trend analysis agent"""
        query = next(m.content for m in reversed(state.messages) if isinstance(m, HumanMessage))
        result = self.trend_agent.analyze(query, self.vector_store)
        return Command(
            update={
                "messages": state.messages + [
                    HumanMessage(content=result, name=self.trend_agent.agent_id)
                ]
            },
            goto=SUPERVISOR_NODE,
        )
    
    def _comparison_node(self, state: MessagesState) -> Command[SupervisorNodeName]:
        """Run comparison analysis agent"""
        query = next(m.content for m in reversed(state.messages) if isinstance(m, HumanMessage))
        result = self.comparison_agent.analyze(query, self.vector_store)
        return Command(
            update={
                "messages": state.messages + [
                    HumanMessage(content=result, name=self.comparison_agent.agent_id)
                ]
            },
            goto=SUPERVISOR_NODE,
        )
    
    def _summary_node(self, state: MessagesState) -> Command[SupervisorNodeName]:
        """Run summary agent"""
        query = next(m.content for m in reversed(state.messages) if isinstance(m, HumanMessage))
        result = self.summary_agent.analyze(query, self.vector_store)
        return Command(
            update={
                "messages": state.messages + [
                    HumanMessage(content=result, name=self.summary_agent.agent_id)
                ]
            },
            goto=SUPERVISOR_NODE,
        )
    
    def analyze(self, query: str) -> List[BaseMessage]:
        """Run analysis with all appropriate agents"""
        # Initialize state with query
        state = MessagesState(messages=[HumanMessage(content=query)])
        
        # Run graph and collect results
        results = []
        current_messages = set()
        
        for output in self.graph.stream(state):
            # Skip end state
            if "__end__" in output:
                continue
                
            # Get state from the current node
            for node_output in output.values():
                if isinstance(node_output, dict) and "messages" in node_output:
                    # Get new messages (skip the query message)
                    new_messages = [
                        msg for msg in node_output["messages"] 
                        if isinstance(msg, AIMessage)
                    ]
                    
                    # Add results with agent names
                    for msg in new_messages:
                        if msg.content not in current_messages:
                            current_messages.add(msg.content)
                            results.append(msg)
        
        return results 