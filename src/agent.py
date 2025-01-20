#!/usr/bin/env python3

import argparse
import logging
from typing import Annotated, Dict, TypedDict
from typing_extensions import TypedDict
from dotenv import load_dotenv
import time

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, Graph, MessageGraph, StateGraph
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.agents import AgentFinish

from src.agents.document_processor import DocumentProcessor
from src.agents.vector_store import VectorStore
from src.agents.insurance_analysis import InsuranceAnalysisAgent
from src.types import AgentState

# Load environment variables from .env
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def supervisor_function(state: AgentState) -> Dict:
    """
    The supervisor function decides which agent should run next based on the current state.
    """
    logger.info(f"Supervisor received state: {state}")
    
    # If we have an input path but no processed documents, process documents first
    if state.input_path and not state.processed_documents:
        logger.info("Starting document processing")
        return {"next": "document_processor"}
        
    # If we have processed documents but no vectors stored, store them next
    if state.processed_documents and not state.vectors_stored:
        logger.info("Starting vector storage")
        return {"next": "vector_store"}
        
    # If we have HumanMessage and stored vectors, run analysis
    if state.messages.count(HumanMessage) > 0 and state.vectors_stored:
        logger.info("Starting analysis")
        return {"next": "analysis"}
        
    logger.info("No more tasks to process")
    return {"next": END}

def create_agent_graph() -> Graph:
    """
    Creates the agent workflow graph.
    """
    # Initialize vector store first
    vector_store = VectorStore()
    vector_store.init_store()
    
    # Initialize agents
    doc_processor = DocumentProcessor(vector_store)
    analysis_agent = InsuranceAnalysisAgent(vector_store)

    # Create the workflow
    workflow = StateGraph(AgentState)

    # Add nodes to the graph
    workflow.add_node("document_processor", doc_processor)
    workflow.add_node("vector_store", vector_store)
    workflow.add_node("analysis", analysis_agent)
    
    # Add the supervisor node
    workflow.add_node("supervisor", supervisor_function)

    # Add edges - define the flow between nodes
    workflow.add_edge("supervisor", "document_processor")

    # Set the entry point
    workflow.set_entry_point("supervisor")

    # Compile the graph
    return workflow.compile()

def main():
    parser = argparse.ArgumentParser(description='Insurance Data Analysis Pipeline')
    parser.add_argument(
        '--input-path', 
        type=str, 
        help='Path to Excel/HTML file or directory containing files'
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
        messages=[HumanMessage(content=args.query if args.query else "Start processing documents")],
        role = "human",
        next = "supervisor",
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
