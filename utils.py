import os
import numpy as np
import requests
from tqdm import tqdm
import fitz  # PyMuPDF
import spacy
from config import Config
import logging
import hashlib
import time
import re

# Initialize spaCy for sentence splitting
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_pdf_hash(file_input) -> str:
    """Generiert einen eindeutigen Hash für die PDF-Datei. Akzeptiert entweder einen Pfad (str) oder einen File-Stream."""
    if isinstance(file_input, str):
        with open(file_input, 'rb') as f:
            file_bytes = f.read()
    else:
        # Annahme: file_input ist ein file-like Objekt
        file_bytes = file_input.read()
        file_input.seek(0)
    return hashlib.md5(file_bytes).hexdigest()


def text_formatter(text: str) -> str:
    """Formats the text by removing line breaks."""
    return text.replace("\n", " ").strip()


def clean_text_chunk(text: str) -> str:
    """
    Cleans text chunks by removing or escaping special characters
    that could be interpreted as commands.
    """
    # Remove control characters
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    
    # Escape special characters that might be interpreted as commands
    text = text.replace('\\', '\\\\')  # Escape backslashes
    text = text.replace('"', '\\"')    # Escape double quotes
    text = text.replace('\n', ' ')     # Replace newlines with spaces
    text = text.replace('\r', ' ')     # Replace carriage returns with spaces
    
    # Remove any remaining non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    
    return text.strip()

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """Reads a PDF file and extracts text with metadata"""
    try:
        doc = fitz.open(pdf_path)
        pages_and_texts = []
        for page_number, page in enumerate(doc):
            text = text_formatter(text=page.get_text())
            pages_and_texts.append({
                "page_number": page_number,
                "text": text,
                "sentences": [str(sentence) for sentence in nlp(text).sents]
            })
        return pages_and_texts
    except Exception as e:
        logging.error(f"Error reading PDF {pdf_path}: {str(e)}")
        raise

def split_list(input_list: list[str], slice_size: int = 10) -> list[list[str]]:
    """Splits a list of sentences into smaller chunks."""
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

def preprocess_and_chunk(pdf_path: str):
    """Loads the PDF, extracts text, and creates chunks."""
    pages_and_texts = open_and_read_pdf(pdf_path)
    pages_and_chunks = []
    
    for item in pages_and_texts:
        item["sentence_chunks"] = split_list(item["sentences"], slice_size=10)
        for chunk in item["sentence_chunks"]:
            chunk_text = " ".join(chunk).strip()
            if chunk_text:  # Only add non-empty chunks
                cleaned_text = clean_text_chunk(chunk_text)  # Clean the text
                pages_and_chunks.append({
                    "page_number": item["page_number"],
                    "sentence_chunk": cleaned_text
                })
    return pages_and_chunks

def cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def load_or_generate_embeddings(text_chunks, pdf_hash: str, progress_callback=None):
    """
    Load existing embeddings or generate new ones for the provided text chunks.
    If a progress_callback is provided, it will be called with (pdf_hash, progress_percentage).
    """
    embeddings_path = os.path.join("embeddings", f"{pdf_hash}.npy")
    
    if os.path.exists(embeddings_path):
        return np.load(embeddings_path)
    
    embeddings = []
    total_batches = (len(text_chunks) + Config.EMBEDDING_BATCH_SIZE - 1) // Config.EMBEDDING_BATCH_SIZE
    
    # Process text chunks in batches
    for batch_index, i in enumerate(range(0, len(text_chunks), Config.EMBEDDING_BATCH_SIZE)):
        batch = text_chunks[i:i + Config.EMBEDDING_BATCH_SIZE]
        logger.debug(f"Processing batch {batch_index + 1}/{total_batches}: {batch}")
        
        cleaned_batch = []
        for chunk in batch:
            if not isinstance(chunk, str) or not chunk.strip():
                logger.warning(f"Skipping invalid chunk: {chunk}")
                continue
            cleaned_chunk = clean_text_chunk(chunk)
            cleaned_batch.append(cleaned_chunk)
        
        if not cleaned_batch:
            logger.warning("Skipping empty batch after cleaning")
            continue
        
        payload = {
            "input": cleaned_batch,
            "model": Config.EMBEDDING_MODEL_NAME
        }
        
        try:
            response = requests.post(
                Config.LM_STUDIO_EMBEDDING_API_URL,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            embedding_data = response.json().get("data", [])
            for item in embedding_data:
                if isinstance(item, dict) and "embedding" in item:
                    embeddings.append(item["embedding"])
                else:
                    logger.error(f"Unexpected embedding format: {item}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to process batch {batch_index + 1}: {str(e)}")
            logger.error(f"Problematic batch: {batch}")
            continue
        
        # Update progress using the callback, if provided
        if progress_callback:
            progress = (batch_index + 1) / total_batches * 100
            progress_callback(pdf_hash, progress)
    
    if not embeddings:
        raise RuntimeError("No embeddings generated for the provided text chunks")
    
    embeddings_array = np.array(embeddings)
    np.save(embeddings_path, embeddings_array)
    return embeddings_array

def make_api_request(url: str, payload: dict, error_msg: str) -> dict:
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"{error_msg}: {str(e)}")
        raise RuntimeError(f"{error_msg}: {str(e)}")