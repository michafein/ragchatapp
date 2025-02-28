"""
Text processing utilities for the RAG chatbot.
Handles text formatting, cleaning, and sentence extraction.
"""

import re
import logging
from typing import List, Optional
import spacy

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize spaCy for sentence splitting
try:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
except Exception as e:
    logger.error(f"Failed to initialize spaCy: {e}")
    # Fallback to a simple regex-based sentence splitter
    def simple_sentence_split(text):
        return re.split(r'(?<=[.!?])\s+', text)
    nlp = None

def text_formatter(text: str) -> str:
    """
    Formats the text by removing line breaks and extra whitespace.
    
    Args:
        text: Raw input text
        
    Returns:
        Formatted text string
    """
    if not isinstance(text, str):
        logger.warning(f"Expected string but got {type(text)}")
        text = str(text)
    
    return text.replace("\n", " ").strip()

def clean_text_chunk(text: str) -> str:
    """
    Cleans text chunks by removing or escaping special characters.
    
    Args:
        text: Raw text chunk
        
    Returns:
        Cleaned text chunk
    """
    if not isinstance(text, str):
        logger.warning(f"Expected string but got {type(text)}")
        text = str(text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    
    # Properly escape special characters - fixed to preserve all backslashes
    text = text.replace('\\', '\\\\')  # First double all backslashes
    text = text.replace('"', '\\"')    # Then escape quotes
    
    # Replace newlines with spaces
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def split_list(input_list: List[str], slice_size: int = 10) -> List[List[str]]:
    """
    Splits a list of sentences into smaller chunks of specified size.
    
    Args:
        input_list: List of strings to split
        slice_size: Maximum size of each chunk
        
    Returns:
        List of lists, where each inner list is a chunk of the original list
    """
    if not input_list:
        return []
    
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

def extract_sentences(text: str) -> List[str]:
    """
    Extracts individual sentences from a text using spaCy.
    
    Args:
        text: Input text to process
        
    Returns:
        List of extracted sentences
    """
    if not text or not isinstance(text, str):
        return []
    
    # Use spaCy if available, otherwise use simple fallback
    if nlp:
        doc = nlp(text)
        return [str(sentence).strip() for sentence in doc.sents if str(sentence).strip()]
    else:
        # Fallback to simple sentence splitter
        return [s.strip() for s in simple_sentence_split(text) if s.strip()]

def ensure_complete_sentences(text: str) -> str:
    """
    Ensures text ends with a complete sentence by truncating at the last sentence-ending punctuation.
    
    Args:
        text: Input text
        
    Returns:
        Text ending with complete sentence
    """
    if not text:
        return ""
    
    # Find the position of the last sentence-ending punctuation
    last_punctuation = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    
    if last_punctuation != -1:
        # Return the text up to and including the punctuation
        return text[:last_punctuation + 1]
    else:
        # If no punctuation found, return the original text
        return text