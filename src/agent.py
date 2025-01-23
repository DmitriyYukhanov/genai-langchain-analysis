#!/usr/bin/env python3

import argparse
import logging
from typing import Annotated, Dict, TypedDict
from typing_extensions import TypedDict, Literal
from dotenv import load_dotenv
import time
from enum import Enum

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import START, END, Graph, MessageGraph, StateGraph
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.agents import AgentFinish
from langgraph.types import Command

from src.agents.document_processor import DocumentProcessor
from src.agents.vector_store import VectorStore
from src.agents.analysis import AnalysisAgent
from src.types import AgentState

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPERVISOR_NODE = "supervisor"

class NodeName(str, Enum):
    DOC_PROCESSOR = DocumentProcessor.NODE_NAME
    VECTOR_STORE = VectorStore.NODE_NAME
    ANALYSIS_AGENT = AnalysisAgent.NODE_NAME

def supervisor_node(state: AgentState) -> Command[NodeName]:
    """
    The supervisor function decides which agent should run next based on the current state.
    """
    logger.info(f"Supervisor received state: {state}")

    if state.error:
        logger.error(f"Error: {state.error}")
        return Command(goto=END)
    
    # If we have an input path but no processed documents, process documents first
    if state.input_path and not state.processed_documents:
        logger.info("Switching to document processing")
        return Command(goto=DocumentProcessor.NODE_NAME)
        
    # If we have processed documents but no vectors stored, store them next
    if state.processed_documents and not state.vectors_stored:
        logger.info("Switching to vector storage")
        return Command(goto=VectorStore.NODE_NAME)
        
    # If we have stored vectors, run analysis
    if state.vectors_stored:
        logger.info("Switching to analysis")
        return Command(goto=AnalysisAgent.NODE_NAME)
        
    logger.info("No more tasks to process")
    return Command(goto=END)

def create_agent_graph() -> Graph:
    """
    Creates the agent workflow graph.
    """

    vector_store = VectorStore()
    doc_processor = DocumentProcessor()
    analysis_agent = AnalysisAgent(vector_store)

    # Create the workflow
    workflow = StateGraph(AgentState)

    workflow.set_entry_point(SUPERVISOR_NODE)
    
    # Add nodes to the graph
    workflow.add_node(SUPERVISOR_NODE, supervisor_node)
    workflow.add_node(DocumentProcessor.NODE_NAME, doc_processor)
    workflow.add_node(VectorStore.NODE_NAME, vector_store)
    workflow.add_node(AnalysisAgent.NODE_NAME, analysis_agent)

    # Add edges to the graph
    workflow.add_edge(DocumentProcessor.NODE_NAME, SUPERVISOR_NODE)
    workflow.add_edge(VectorStore.NODE_NAME, SUPERVISOR_NODE)
    workflow.add_edge(AnalysisAgent.NODE_NAME, SUPERVISOR_NODE)

    # Compile the graph
    return workflow.compile()

def main():
    parser = argparse.ArgumentParser(description='Insurance Data Analysis Pipeline')
    parser.add_argument(
        '--input-path', 
        type=str, 
        help='Path to Excel/HTML file or directory containing files',
        required=True
    )
    parser.add_argument(
        '--query', 
        type=str, 
        help='Analysis query to run'
    )
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Enable debug logging'
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create the agent graph
    graph = create_agent_graph()

    initial_state = AgentState(
        messages=[HumanMessage(content=args.query if args.query else "Analyze documents and tell any interesting findings, insights, or highlight you can find out.")],
        role = "human",
        input_path = args.input_path,
        processed_documents = None,
        vectors_stored = False
    )

    # Measure total time
    start_time = time.time()

    # Run the graph
    for output in graph.stream(initial_state):
        if "__end__" not in output:
            if output.get("processed_documents"):
                docs = output["processed_documents"]
                logger.info(f"Successfully processed {len(docs)} document chunks")
                if docs:
                    logger.info("\nSample from first document:")
                    logger.info(f"Content: {docs[0].page_content[:200]}...")
                    logger.info(f"Metadata: {docs[0].metadata}")
            else:
                logger.info(f"Intermediate output: {output}")

    # Calculate and log total time
    total_time = time.time() - start_time
    logger.info(f"Total analysis time: {total_time:.2f} seconds")
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()
