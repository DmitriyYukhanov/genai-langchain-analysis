import os
import logging

logger = logging.getLogger(__name__)

def detect_file_type(file_path: str) -> str:
    """
    Detect if file is XLS or HTML format
    
    Args:
        file_path: Path to the file
        
    Returns:
        String indicating file type ('xls' or 'html')
    """
    # First check extension
    ext = os.path.splitext(file_path)[1].lower()
    
    # Check content only if extension is ambiguous
    if ext not in ['.html', '.htm']:
        try:
            with open(file_path, 'rb') as f:
                header = f.read(10).strip()
                # Convert to string for easier pattern matching, ignore decode errors
                header_str = header.decode('utf-8', errors='ignore').lower()
                
                # Look for HTML signatures
                html_markers = [
                    b'<!doctype html',
                    b'<html',
                    b'<head',
                    b'<body',
                    b'<table'
                ]
                
                # Check binary patterns
                if any(marker in header.lower() for marker in html_markers):
                    return 'html'
                    
                # Additional check for string patterns (handles whitespace/BOM cases)
                if any(marker.decode().lower() in header_str for marker in html_markers):
                    return 'html'
                    
        except Exception as e:
            logger.debug(f"Error reading file header: {str(e)}")
            # If we can't read the file, fall back to extension-based detection
            pass
    
    # Return based on extension or default to XLS
    if ext in ['.html', '.htm']:
        return 'html'
    elif ext in ['.xls', '.xlsx']:
        return 'xls'
    return 'xls' 