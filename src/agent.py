#!/usr/bin/env python3

import argparse
import logging
from typing import Annotated, Dict, TypedDict
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, Graph, MessageGraph
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.agents import AgentFinish

from agents.document_processor import DocumentProcessor
from agents.vector_store import VectorStore
from agents.insurance_analysis import InsuranceAnalysisAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: list[BaseMessage]
    next: str

def supervisor_function(state: AgentState) -> Dict:
    """
    The supervisor function decides which agent should run next based on the current state.
    This is where you'll implement the logic for agent coordination.
    """
    # TODO: Implement the supervisor logic
    # Example logic:
    # - Check if documents need processing
    # - Check if vector store needs updating
    # - Check if analysis is needed
    return {"next": END}

def create_agent_graph() -> Graph:
    """
    Creates the agent workflow graph.
    """
    # Initialize agents
    doc_processor = DocumentProcessor()
    vector_store = VectorStore()
    analysis_agent = InsuranceAnalysisAgent(vector_store)

    # Create the workflow
    workflow = MessageGraph()

    # Add nodes to the graph
    # TODO: Add the actual agent nodes and their functions

    # Add the supervisor node
    workflow.add_node("supervisor", supervisor_function)

    # Set the entry point
    workflow.set_entry_point("supervisor")

    # Compile the graph
    return workflow.compile()

def main():
    parser = argparse.ArgumentParser(description='Insurance Data Analysis Pipeline')
    parser.add_argument('--excel-path', type=str, help='Path to Excel file for processing')
    parser.add_argument('--query', type=str, help='Analysis query to run')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create the agent graph
    graph = create_agent_graph()

    # Initialize the state
    initial_state = AgentState(
        messages=[HumanMessage(content=args.query)] if args.query else [],
        next="supervisor"
    )

    # Run the graph
    for output in graph.stream(initial_state):
        if "__end__" not in output:
            logger.info(f"Intermediate output: {output}")

    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()
