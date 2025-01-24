import time
import logging
import functools
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

class TimingStats:
    """Tracks timing statistics for the application"""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.step_times: Dict[str, float] = {}
    
    def start(self) -> None:
        """Start tracking time"""
        self.start_time = time.time()
    
    def record_step(self, step_name: str, duration: float) -> None:
        """Record the duration of a step"""
        self.step_times[step_name] = duration
    
    @property
    def total_time(self) -> float:
        """Get total elapsed time since start"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

# Global stats tracker
_stats = TimingStats()

def measure_time(func: Callable) -> Callable:
    """
    Decorator to measure execution time of functions.
    Logs the time taken if it's more than 1 second.
    """
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Start tracking if this is the first timed function
        if _stats.start_time is None:
            _stats.start()
        
        # Measure function execution time
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        # Get function name without class prefix
        func_name = func.__name__.split('.')[-1]
        
        # Record step timing
        _stats.record_step(func_name, elapsed_time)
        
        # Log timing if significant
        if elapsed_time > 1:  # Only log if took more than a second
            if logger.getEffectiveLevel() <= logging.DEBUG:
                # In debug mode, show all timing information
                logger.debug(
                    f"{func_name} took {elapsed_time:.2f}s "
                    f"(Total: {_stats.total_time:.2f}s)"
                )
            else:
                # In non-debug mode, just show the timing
                logger.info(f"{func_name} took {elapsed_time:.1f}s")
        
        return result
        
    return wrapper 