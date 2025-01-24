from typing import Optional, Any, Dict, TypeVar, List, Callable
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TaskID
)
from rich.console import Console
from rich.logging import RichHandler
from rich.live import Live
import logging
from functools import wraps
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

console = Console()

T = TypeVar('T')
R = TypeVar('R')

class ProgressManager:
    """Manages progress bars and status updates for long-running operations"""
    
    _instance = None
    _is_running = False
    
    def __new__(cls, *args, **kwargs):
        """Ensure only one instance exists"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize progress manager"""
        if not hasattr(self, 'progress'):
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(complete_style="green", finished_style="bright_green"),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                TextColumn("[bold]{task.completed}/{task.total}"),
                console=console,
                expand=True,
                transient=False,  # Keep progress bars visible
                refresh_per_second=10
            )
            self.tasks = {}
            self._live = None

    def __enter__(self):
        """Start progress tracking"""
        if not self._is_running:
            self._live = Live(
                self.progress,
                refresh_per_second=10,
                transient=False  # Keep progress bars visible
            )
            self._live.__enter__()
            self._is_running = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop progress tracking"""
        if self._is_running and not self.tasks:
            self._is_running = False
            self._live.__exit__(exc_type, exc_val, exc_tb)
            self._live = None

    def add_task(self, description: str, total_steps: Optional[int] = None) -> str:
        """Add a new task and return its ID"""
        task_id = self.progress.add_task(
            description,
            total=None if total_steps is None else float(total_steps),
            completed=0.0,  # Explicitly set initial completion
            start=True  # Ensure task starts immediately
        )
        task_key = str(task_id)
        self.tasks[task_key] = {
            'id': task_id,
            'total': total_steps,
            'completed': 0
        }
        return task_key

    def update_task(self, task_key: str, advance: int = 1, description: Optional[str] = None, completed: Optional[float] = None):
        """Update task progress
        
        Args:
            task_key: Task identifier
            advance: Number of steps to advance
            description: New description for the task
            completed: Set absolute completion value
        """
        if task_key in self.tasks:
            task = self.tasks[task_key]
            update_kwargs = {}
            
            if description:
                update_kwargs['description'] = description
            
            if completed is not None:
                # Set absolute completion value
                task['completed'] = min(float(completed), task['total'] if task['total'] is not None else float(completed))
                update_kwargs['completed'] = task['completed']
            elif task['total'] is not None and advance > 0:
                # Advance by specified amount
                task['completed'] = min(task['completed'] + advance, task['total'])
                update_kwargs['completed'] = float(task['completed'])
                
            if update_kwargs:
                self.progress.update(task['id'], **update_kwargs)

    def remove_task(self, task_key: str):
        """Remove a task after ensuring it shows as complete"""
        if task_key in self.tasks:
            task = self.tasks[task_key]
            # Ensure task shows as complete before removal
            if task['total'] is not None:
                self.progress.update(
                    task['id'],
                    completed=float(task['total']),
                    refresh=True
                )
                # Small delay to ensure the update is visible
                time.sleep(0.1)
            self.progress.remove_task(task['id'])
            del self.tasks[task_key]

def setup_logging(debug: bool = False):
    """Setup logging with rich output"""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(
            console=console,
            show_time=False,
            show_path=debug,
            rich_tracebacks=True,
            markup=True,
            show_level=debug  # Only show log levels in debug mode
        )]
    )

def parallel_process(
    items: List[T],
    process_func: Callable[[T], R],
    description: Optional[str] = None,
    max_workers: Optional[int] = None
) -> List[R]:
    """
    Process items in parallel with progress tracking
    
    Args:
        items: List of items to process
        process_func: Function to process each item
        description: Description for progress bar (if None, no progress bar is shown)
        max_workers: Maximum number of worker threads
        
    Returns:
        List of processed results
    """
    results = []
    total_items = len(items)
    task_key = None
    
    with ProgressManager() as progress:
        # Only create progress bar if description is provided
        if description:
            task_key = progress.add_task(description, total_steps=total_items)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(process_func, item): item 
                for item in items
            }
            
            # Process completed tasks
            completed = 0
            
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    if result:  # Check if result is not empty
                        results.append(result)
                    completed += 1
                    
                    # Update progress only if we have a progress bar
                    if task_key and description:
                        progress.update_task(
                            task_key,
                            completed=float(completed),
                            description=f"{description} ({completed}/{total_items})"
                        )
                except Exception as e:
                    completed += 1
                    logging.error(f"Error processing {item}: {str(e)}")
                    if task_key and description:
                        progress.update_task(
                            task_key,
                            completed=float(completed),
                            description=f"{description} ({completed}/{total_items})"
                        )
            
            # Remove progress bar if we created one
            if task_key:
                # Ensure 100% completion
                progress.update_task(
                    task_key,
                    completed=float(total_items),
                    description=f"{description} - Complete"
                )
                progress.remove_task(task_key)
                    
    return results 