import os
import logging
import numpy as np
import requests
from typing import List, Dict, Any, Callable, Optional
from functools import lru_cache

# Import from config
from config import Config

# Initialize logging
logger = logging.getLogger(__name__)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    # Check for zero vectors to avoid division by zero
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def make_api_request(url: str, payload: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
    """
    Makes an API request and returns the JSON response.
    
    Args:
        url: API endpoint URL
        payload: Request payload
        error_msg: Error message prefix for failures
        
    Returns:
        JSON response data
    """
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"{error_msg}: {str(e)}")
        raise RuntimeError(f"{error_msg}: {str(e)}")

def load_or_generate_embeddings(text_chunks: List[str], pdf_hash: str, 
                               progress_callback: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """
    Loads existing embeddings from disk or generates new ones if not available.
    
    Args:
        text_chunks: List of text chunks to embed
        pdf_hash: Hash identifier for the PDF
        progress_callback: Optional callback function for progress updates
        
    Returns:
        NumPy array of embeddings
    """
    # Make sure embeddings directory exists
    os.makedirs("embeddings", exist_ok=True)
    
    embeddings_path = os.path.join("embeddings", f"{pdf_hash}.npy")
    
    # If embeddings already exist, load them
    if os.path.exists(embeddings_path):
        logger.info(f"Loading existing embeddings for {pdf_hash}")
        return np.load(embeddings_path)
    
    logger.info(f"Generating new embeddings for {pdf_hash}")
    
    # Generate embeddings
    embeddings = []
    total = len(text_chunks)
    batch_size = Config.EMBEDDING_BATCH_SIZE
    
    for i in range(0, total, batch_size):
        batch = text_chunks[i:i + batch_size]
        logger.debug(f"Processing batch starting at index {i}: {len(batch)} chunks")
        
        cleaned_batch = []
        for chunk in batch:
            if not isinstance(chunk, str) or not chunk.strip():
                logger.warning(f"Skipping invalid chunk: {chunk}")
                continue
            cleaned_batch.append(chunk.strip())
            
        if not cleaned_batch:
            logger.warning("Skipping empty batch after cleaning")
            continue
            
        payload = {
            "input": cleaned_batch,
            "model": Config.EMBEDDING_MODEL_NAME
        }
        
        try:
            # Make API request for embeddings
            response = requests.post(
                Config.LM_STUDIO_EMBEDDING_API_URL,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            # Extract embedding data
            embedding_data = response.json().get("data", [])
            
            for item in embedding_data:
                if isinstance(item, dict) and "embedding" in item:
                    embeddings.append(item["embedding"])
                else:
                    logger.error(f"Unexpected embedding format: {item}")
                    continue
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to process batch starting at index {i}: {str(e)}")
            logger.error(f"Problematic batch: {batch}")
            continue
            
        # Report progress if callback provided
        if progress_callback:
            current = min(i + batch_size, total)
            progress_callback(current, total)
    
    # Check if we got any embeddings
    if not embeddings:
        logger.error("No embeddings were generated")
        raise RuntimeError("Failed to generate any embeddings")
    
    # Convert to numpy array and save
    embeddings_array = np.array(embeddings)
    
    # Save embeddings with explicit absolute path
    logger.info(f"Saving embeddings to {embeddings_path}")
    try:
        np.save(embeddings_path, embeddings_array)
        logger.info(f"Successfully saved embeddings with shape {embeddings_array.shape}")
    except Exception as e:
        logger.error(f"Error saving embeddings: {e}")
        logger.error(f"Tried to save to: {embeddings_path}")
        raise RuntimeError(f"Failed to save embeddings: {e}")
    
    return embeddings_array

@lru_cache(maxsize=5)
def get_all_embeddings() -> tuple:
    """
    Loads all embeddings and metadata from storage with caching.
    
    Returns:
        Tuple of (embeddings array, metadata list)
    """
    all_embeddings = []
    all_metadata = []
    embeddings_dir = "embeddings"
    pages_dir = "pages_and_chunks"
    
    if os.path.exists(embeddings_dir) and os.path.isdir(embeddings_dir):
        for file in os.listdir(embeddings_dir):
            if file.endswith(".npy"):
                pdf_hash = file.split(".")[0]
                try:
                    embeddings = np.load(os.path.join(embeddings_dir, file))
                    
                    metadata_path = os.path.join(pages_dir, f"{pdf_hash}.json")
                    if os.path.exists(metadata_path):
                        import json
                        with open(metadata_path, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                            
                            # Add PDF name to each chunk for reference
                            for chunk in metadata["chunks"]:
                                chunk["pdf_hash"] = pdf_hash
                                chunk["pdf_name"] = metadata.get("pdf_name", "Unknown PDF")
                                
                            all_metadata.extend(metadata["chunks"])
                            all_embeddings.append(embeddings)
                    else:
                        logger.warning(f"Metadata not found for {pdf_hash}")
                except Exception as e:
                    logger.error(f"Error processing {file}: {str(e)}")
                    continue
    
    if not all_embeddings:
        logger.warning("No embeddings found")
        return np.array([]), []
    
    return np.concatenate(all_embeddings), all_metadata

def clear_embeddings_cache():
    """Clears the embeddings cache"""
    get_all_embeddings.cache_clear()