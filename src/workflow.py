#!/usr/bin/env python3

import argparse
import logging
from typing import Dict
from dotenv import load_dotenv
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, Graph, StateGraph
from langgraph.types import Command

from src.systems.document_processor import DocumentProcessor
from src.systems.vector_store import VectorStore
from src.systems.analysis.analysis_graph import AnalysisAgent
from src.systems.types import SystemState, WorkflowNode, Status
from src.utils.progress import setup_logging, ProgressManager

load_dotenv()

logger = logging.getLogger(__name__)

SUPERVISOR_NODE = "supervisor"

# Initialize workflow nodes
doc_processor = DocumentProcessor()
vector_store = VectorStore()
analysis_agent = AnalysisAgent(vector_store)

class NodeName(str, Enum):
    DOC_PROCESSOR = doc_processor.NODE_NAME
    VECTOR_STORE = vector_store.NODE_NAME
    ANALYSIS_AGENT = analysis_agent.NODE_NAME

class NodeTracker:
    """Tracks workflow node execution"""
    def __init__(self, nodes: Dict[str, WorkflowNode]):
        self.nodes = nodes
        self.completed_nodes = set()
        self.total_nodes = len(nodes)
        self.current_node = None
    
    def start_node(self, node_name: str) -> None:
        """Track start of node execution"""
        self.current_node = node_name
    
    def complete_node(self, node_name: str) -> None:
        """Track completion of node execution"""
        self.completed_nodes.add(node_name)
    
    @property
    def progress(self) -> float:
        """Get current progress as float between 0 and 1"""
        return len(self.completed_nodes) / self.total_nodes
    
    def get_next_description(self) -> str:
        """Get description for the next node"""
        if not self.current_node or self.current_node not in self.nodes:
            return "Analysis Pipeline"
        return f"Analysis Pipeline - {self.nodes[self.current_node].STEP_DESCRIPTION}"

def supervisor_node(state: SystemState, tracker: NodeTracker) -> Command[NodeName]:
    """
    The supervisor function decides which agent should run next based on the current state.
    """
    try:
        # Create a concise state summary
        state_summary = {
            "status": state.status,
            "has_error": bool(state.error),
            "has_input": bool(state.input_path)
        }
        logger.debug("Supervisor state: %s", state_summary)

        # Handle errors first
        if state.error:
            state.status = Status.ERROR
            logger.error(f"Error encountered: {state.error}")
            return Command(goto=END)
        
        # Determine next step based on pipeline status
        if state.status == Status.INIT and state.input_path:
            state.status = Status.DOCS_PROCESSING
            logger.debug("Starting document processing")
            tracker.start_node(doc_processor.NODE_NAME)
            return Command(goto=doc_processor.NODE_NAME)
            
        if state.status == Status.DOCS_PROCESSED:
            state.status = Status.VECTORS_SAVING
            logger.debug("Starting vector storage")
            # Mark previous node as complete in tracker
            if tracker.current_node:
                tracker.complete_node(tracker.current_node)
            tracker.start_node(vector_store.NODE_NAME)
            return Command(goto=vector_store.NODE_NAME)
            
        if state.status == Status.VECTORS_SAVED:
            state.status = Status.ANALYSIS_RUNNING
            logger.debug("Starting extracted data analysis")
            # Mark previous node as complete in tracker
            if tracker.current_node:
                tracker.complete_node(tracker.current_node)
            tracker.start_node(analysis_agent.NODE_NAME)
            return Command(goto=analysis_agent.NODE_NAME)
            
        # If we reach here, we're done
        # Mark last node as complete in tracker
        if tracker.current_node:
            tracker.complete_node(tracker.current_node)
        state.status = Status.DONE
        logger.debug("Workflow complete")
        return Command(goto=END)

    except Exception as e:
        logger.error(f"Error in supervisor: {str(e)}")
        state.error = f"Supervisor error: {str(e)}"
        state.status = Status.ERROR
        return Command(goto=END)

def create_agent_graph(tracker: NodeTracker) -> Graph:
    """
    Creates the agent workflow graph with dynamic routing based on agent capabilities.
    """
    # Create the workflow
    workflow = StateGraph(SystemState)

    workflow.set_entry_point(SUPERVISOR_NODE)
    
    # Add nodes to the graph
    workflow.add_node(SUPERVISOR_NODE, lambda x: supervisor_node(x, tracker))
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
            
        # Map node names to their instances
        workflow_nodes = {
            doc_processor.NODE_NAME: doc_processor,
            vector_store.NODE_NAME: vector_store,
            analysis_agent.NODE_NAME: analysis_agent
        }
        
        # Initialize node tracker
        node_tracker = NodeTracker(workflow_nodes)
        
        # Create the agent graph with node tracker
        graph = create_agent_graph(node_tracker)

        # Initialize state
        initial_state = SystemState(
            messages=[HumanMessage(content=args.query)],
            role="human",
            input_path=args.input_path,
            processed_documents=None,
            vectors_stored=False,
            error=None
        )
        
        # Run the graph with progress tracking
        previous_messages = set()
        
        with ProgressManager() as progress:
            pipeline_task = progress.add_task(
                description="Analysis Pipeline",
                total_steps=node_tracker.total_nodes
            )
            
            for output in graph.stream(initial_state):
                if "__end__" not in output:
                    if "error" in output and output["error"]:
                        logger.error(f"Error in pipeline: {output['error']}")
                        break
                    
                    # Update progress based on node tracker
                    progress.update_task(
                        pipeline_task,
                        completed=float(len(node_tracker.completed_nodes)),
                        description=node_tracker.get_next_description()
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
                                # Print analysis results with minimal formatting
                                logger.info("\n" + content)
                        
                        previous_messages = current_messages
            
            # Ensure progress bar shows completion
            progress.update_task(
                pipeline_task,
                completed=float(node_tracker.total_nodes),
                description="Analysis Pipeline - Complete"
            )
            progress.remove_task(pipeline_task)

        if args.debug:
            logger.info("Analysis complete!")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
