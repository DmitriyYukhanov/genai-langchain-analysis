import time
from functools import wraps
import logging

logger = logging.getLogger(__name__)

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"{func.__name__} took {elapsed_time:.2f} seconds")
        return result
    return wrapper 