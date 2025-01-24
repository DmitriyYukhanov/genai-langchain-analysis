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

from src.systems.document_processor import DocumentProcessor
from src.systems.vector_store import VectorStore
from src.systems.analysis_graph import AnalysisAgent
from src.systems.types import SystemState
from src.utils.progress import setup_logging, ProgressManager

load_dotenv()

logger = logging.getLogger(__name__)

SUPERVISOR_NODE = "supervisor"

class NodeName(str, Enum):
    DOC_PROCESSOR = DocumentProcessor().NODE_NAME
    VECTOR_STORE = VectorStore().NODE_NAME
    ANALYSIS_AGENT = AnalysisAgent(VectorStore()).NODE_NAME

def supervisor_node(state: SystemState) -> Command[NodeName]:
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
        logger.debug("Supervisor state: %s", state_summary)

        # Handle errors first
        if state.error:
            logger.error(f"Error encountered: {state.error}")
            return Command(goto=END)
        
        # Get node instances for name lookup
        doc_processor = DocumentProcessor()
        vector_store = VectorStore()
        analysis_agent = AnalysisAgent(vector_store)
        
        # Determine next step based on state
        if state.input_path and not state.processed_documents:
            logger.debug("Starting document processing")
            return Command(goto=doc_processor.NODE_NAME)
            
        if state.processed_documents and not state.vectors_stored:
            logger.debug("Starting vector storage")
            return Command(goto=vector_store.NODE_NAME)
            
        # Check if analysis is needed
        if state.vectors_stored and state.messages:
            # Get last two messages
            last_messages = state.messages[-2:] if len(state.messages) >= 2 else state.messages
            
            # If last message is human and not followed by AI response, run analysis
            if any(isinstance(msg, HumanMessage) for msg in last_messages) and not any(isinstance(msg, AIMessage) for msg in last_messages):
                logger.debug("Starting extracted data analysis")
                return Command(goto=analysis_agent.NODE_NAME)
            
        # If we reach here, we're done
        logger.debug("Workflow complete")
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
    doc_processor = DocumentProcessor()
    vector_store = VectorStore()
    analysis_agent = AnalysisAgent(vector_store)
    
    # Create the workflow
    workflow = StateGraph(SystemState)

    workflow.set_entry_point(SUPERVISOR_NODE)
    
    # Add nodes to the graph
    workflow.add_node(SUPERVISOR_NODE, supervisor_node)
    workflow.add_node(doc_processor.NODE_NAME, doc_processor)
    workflow.add_node(vector_store.NODE_NAME, vector_store)
    workflow.add_node(analysis_agent.NODE_NAME, analysis_agent)

    # Add edges to the graph
    workflow.add_edge(doc_processor.NODE_NAME, SUPERVISOR_NODE)
    workflow.add_edge(vector_store.NODE_NAME, SUPERVISOR_NODE)
    workflow.add_edge(analysis_agent.NODE_NAME, SUPERVISOR_NODE)

    # Compile the graph
    return workflow.compile()

def main():
    """Main entry point for the analysis pipeline"""
    try:
        parser = argparse.ArgumentParser(description='Data Analysis Pipeline')
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

        # Setup logging with rich output
        setup_logging(args.debug)

        if args.debug:
            logger.debug("Initializing analysis pipeline...")
            logger.debug(f"Input path: {args.input_path}")
            logger.debug(f"Query: {args.query}")
        else:
            logger.info("Starting analysis...")

        # Create the agent graph
        graph = create_agent_graph()

        # Initialize state
        initial_state = SystemState(
            messages=[HumanMessage(content=args.query)],
            role="human",
            input_path=args.input_path,
            processed_documents=None,
            vectors_stored=False,
            error=None
        )

        # Get workflow nodes for progress tracking
        doc_processor = DocumentProcessor()
        vector_store = VectorStore()
        analysis_agent = AnalysisAgent(vector_store)
        
        # Map node names to their descriptions
        workflow_nodes = {
            doc_processor.NODE_NAME: doc_processor,
            vector_store.NODE_NAME: vector_store,
            analysis_agent.NODE_NAME: analysis_agent
        }
        total_steps = len(workflow_nodes)
        current_step = 0
        
        # Run the graph with progress tracking
        previous_messages = set()
        
        with ProgressManager() as progress:
            pipeline_task = progress.add_task(
                description="Analysis Pipeline",
                total_steps=total_steps
            )
            
            for output in graph.stream(initial_state):
                if "__end__" not in output:
                    if "error" in output and output["error"]:
                        logger.error(f"Error in pipeline: {output['error']}")
                        break
                        
                    # Track workflow progress
                    if output.get("processed_documents") and current_step == 0:
                        current_step += 1
                        next_node = workflow_nodes[vector_store.NODE_NAME]
                        progress.update_task(
                            pipeline_task,
                            completed=float(current_step),  # Use absolute completion
                            description=f"Analysis Pipeline - {next_node.STEP_DESCRIPTION}"
                        )
                    elif output.get("vectors_stored") and current_step == 1:
                        current_step += 1
                        next_node = workflow_nodes[analysis_agent.NODE_NAME]
                        progress.update_task(
                            pipeline_task,
                            completed=float(current_step),  # Use absolute completion
                            description=f"Analysis Pipeline - {next_node.STEP_DESCRIPTION}"
                        )
                    
                    # Handle document processing results
                    if output.get("processed_documents"):
                        docs = output["processed_documents"]
                        if not docs:
                            logger.error("No documents were processed successfully")
                            break
                        if logger.getEffectiveLevel() <= logging.DEBUG:
                            logger.debug(f"Processed {len(docs)} document chunks")
                    
                    # Print new AI messages
                    if output.get("messages"):
                        current_messages = set(msg.content for msg in output["messages"] if isinstance(msg, AIMessage))
                        new_messages = current_messages - previous_messages
                        
                        for content in new_messages:
                            # Print analysis results without any decorations in non-debug mode
                            if args.debug:
                                logger.info("\nAnalysis Result:")
                                logger.info("=" * 80)
                                logger.info(content)
                                logger.info("=" * 80 + "\n")
                            else:
                                logger.info("\n" + content + "\n")
                        
                        previous_messages = current_messages
            
            # Ensure progress bar is filled at the end
            if current_step < total_steps:
                progress.update_task(
                    pipeline_task,
                    advance=0,  # Don't advance, just update completion
                    description="Analysis Pipeline - Complete",
                    completed=float(total_steps)  # Set absolute completion
                )
            progress.remove_task(pipeline_task)

        if args.debug:
            logger.info("Analysis complete!")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
