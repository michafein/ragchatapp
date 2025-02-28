"""
Security utilities for the RAG chatbot.
Handles input validation, sanitization, and security measures.
"""

import re
import logging
import os
from typing import Dict, Any, Optional

# Initialize logging
logger = logging.getLogger(__name__)

def validate_file_upload(file: Any) -> bool:
    """
    Validates that an uploaded file meets security requirements.
    
    Args:
        file: Uploaded file object
        
    Returns:
        Boolean indicating whether the file is valid
    """
    if file is None or not hasattr(file, 'filename') or not file.filename:
        logger.warning("Missing or invalid file")
        return False
        
    # Check if file is a PDF
    if not file.filename.lower().endswith('.pdf'):
        logger.warning(f"Rejected non-PDF file: {file.filename}")
        return False
    
    # Check file size (10MB limit)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    
    max_size = 10 * 1024 * 1024  # 10MB in bytes
    if file_size > max_size:
        logger.warning(f"File too large: {file_size} bytes (max: {max_size})")
        return False
        
    return True

def sanitize_input(text: str) -> str:
    """
    Sanitizes user input to prevent injection attacks.
    
    Args:
        text: User-provided text input
        
    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        return ""
    
    # Pattern to remove scripts
    text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Pattern to remove event handlers but keep the tags
    # This regex looks for any HTML tag with an event handler
    text = re.sub(r'(<[^>]*?)\s+on\w+\s*=\s*["\'][^"\']*?["\']([^>]*?>)', r'\1\2', text, flags=re.IGNORECASE)
    
    # Pattern to remove javascript: protocol
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    
    # Pattern to remove data: URIs
    text = re.sub(r'data:[^,]*?base64', '', text, flags=re.IGNORECASE)
    
    return text.strip()

def set_secure_headers(response: Any) -> Any:
    """
    Sets security headers on HTTP responses.
    
    Args:
        response: Flask response object
        
    Returns:
        Response with security headers
    """
    # Content Security Policy - Restrict resources 
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' https://code.jquery.com/ https://cdn.jsdelivr.net/; style-src 'self' https://cdn.jsdelivr.net/ https://stackpath.bootstrapcdn.com/ 'unsafe-inline'; img-src 'self' data: https://i.ibb.co/; font-src 'self' https://cdnjs.cloudflare.com/ https://use.fontawesome.com/;"
    
    # Prevent browsers from MIME-sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    # Protect against clickjacking
    response.headers['X-Frame-Options'] = 'DENY'
    
    # Enable XSS protection in browsers
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    return response

def validate_api_request(data: Dict[str, Any], required_fields: Optional[list] = None) -> bool:
    """
    Validates that an API request contains required fields and meets security requirements.
    
    Args:
        data: Request data dictionary
        required_fields: List of fields that must be present
        
    Returns:
        Boolean indicating whether the request is valid
    """
    if not isinstance(data, dict):
        logger.warning("Invalid request data format")
        return False
        
    # Check required fields
    if required_fields:
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.warning(f"Missing required fields: {', '.join(missing_fields)}")
            return False
    
    # Check for suspicious content
    for key, value in data.items():
        if isinstance(value, str) and any(pattern in value.lower() for pattern in [
            'javascript:',
            '<script',
            'onerror=',
            'onload=',
            'eval(',
            'document.cookie'
        ]):
            logger.warning(f"Suspicious content detected in field '{key}'")
            return False
    
    return True