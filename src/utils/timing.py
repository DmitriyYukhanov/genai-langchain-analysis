import time
import logging
import functools
from typing import Any, Callable

logger = logging.getLogger(__name__)

def measure_time(func: Callable) -> Callable:
    """Decorator to measure execution time of functions"""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        # Get function name without class prefix if it exists
        func_name = func.__name__.split('.')[-1]
        
        # Make function name more readable
        readable_name = ' '.join(func_name.split('_')).capitalize()
        
        if elapsed_time > 1:  # Only log if took more than a second
            if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug(f"{readable_name} took {elapsed_time:.2f} seconds")
            else:
                # For non-debug mode, only show user-friendly progress messages
                if func_name == 'add_documents':
                    logger.info(f"Processed document embeddings in {elapsed_time:.1f}s")
        
        return result
        
    return wrapper 