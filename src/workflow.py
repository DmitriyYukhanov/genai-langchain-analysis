#!/usr/bin/env python3

import argparse
import logging
from typing import Annotated, Dict, TypedDict
from typing_extensions import TypedDict, Literal
from dotenv import load_dotenv
import time
from enum import Enum

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
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
    try:
        # Create a concise state summary
        state_summary = {
            "has_error": bool(state.error),
            "has_input": bool(state.input_path),
            "docs_processed": bool(state.processed_documents),
            "vectors_stored": state.vectors_stored,
            "has_messages": bool(state.messages)
        }
        logger.info("Supervisor state: %s", state_summary)

        # Handle errors first
        if state.error:
            logger.error(f"Error encountered: {state.error}")
            return Command(goto=END)
        
        # Determine next step based on state
        if state.input_path and not state.processed_documents:
            logger.info("Starting document processing")
            return Command(goto=DocumentProcessor.NODE_NAME)
            
        if state.processed_documents and not state.vectors_stored:
            logger.info("Starting vector storage")
            return Command(goto=VectorStore.NODE_NAME)
            
        # Check if analysis is needed
        if state.vectors_stored and state.messages:
            # Get last two messages
            last_messages = state.messages[-2:] if len(state.messages) >= 2 else state.messages
            
            # If last message is human and not followed by AI response, run analysis
            if any(isinstance(msg, HumanMessage) for msg in last_messages) and not any(isinstance(msg, AIMessage) for msg in last_messages):
                logger.info("Starting analysis")
                return Command(goto=AnalysisAgent.NODE_NAME)
            
        # If we reach here, we're done
        logger.info("Workflow complete")
        return Command(goto=END)

    except Exception as e:
        logger.error(f"Error in supervisor: {str(e)}")
        state.error = f"Supervisor error: {str(e)}"
        return Command(goto=END)

def create_agent_graph() -> Graph:
    """
    Creates the agent workflow graph with dynamic routing based on agent capabilities.
    """
    # Initialize core components
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
    """Main entry point for the analysis pipeline"""
    try:
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
            help='Analysis query to run',
            default="Analyze documents and tell any interesting findings, insights, or highlights you can find."
        )
        parser.add_argument(
            '--debug', 
            action='store_true', 
            help='Enable debug logging'
        )

        args = parser.parse_args()

        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)

        logger.info("Initializing analysis pipeline...")
        logger.info(f"Input path: {args.input_path}")
        logger.info(f"Query: {args.query}")

        # Create the agent graph
        graph = create_agent_graph()

        # Initialize state
        initial_state = AgentState(
            messages=[HumanMessage(content=args.query)],
            role="human",
            input_path=args.input_path,
            processed_documents=None,
            vectors_stored=False,
            error=None
        )

        # Measure total time
        start_time = time.time()

        # Run the graph with progress tracking
        logger.info("Starting analysis...")
        previous_messages = set()
        
        for output in graph.stream(initial_state):
            if "__end__" not in output:
                if "error" in output and output["error"]:
                    logger.error(f"Error in pipeline: {output['error']}")
                    break
                    
                if output.get("processed_documents"):
                    docs = output["processed_documents"]
                    logger.info(f"Successfully processed {len(docs)} document chunks")
                
                # Print new AI messages
                if output.get("messages"):
                    current_messages = set(msg.content for msg in output["messages"] if isinstance(msg, AIMessage))
                    new_messages = current_messages - previous_messages
                    
                    for content in new_messages:
                        logger.info("\nAnalysis Result:")
                        logger.info("=" * 80)
                        logger.info(content)
                        logger.info("=" * 80 + "\n")
                    
                    previous_messages = current_messages

        # Calculate and log total time
        total_time = time.time() - start_time
        logger.info(f"Total analysis time: {total_time:.2f} seconds")
        logger.info("Analysis complete!")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
