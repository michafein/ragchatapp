import os
import numpy as np
import requests
from tqdm import tqdm
import fitz  # PyMuPDF
import spacy
from config import Config
import logging
import hashlib
import re
from typing import Union, List, Dict, Any, Callable

# Initialize spaCy for sentence splitting
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_pdf_hash(file_input: Union[str, Any]) -> str:
    """
    Generates a unique hash for the PDF file.
    Accepts a file path (str) or a file-like object.
    """
    if isinstance(file_input, str):
        with open(file_input, 'rb') as f:
            file_bytes = f.read()
    else:
        file_bytes = file_input.read()
        file_input.seek(0)
    return hashlib.md5(file_bytes).hexdigest()

def text_formatter(text: str) -> str:
    """
    Formats the text by removing line breaks.
    """
    return text.replace("\n", " ").strip()

def clean_text_chunk(text: str) -> str:
    """
    Cleans text chunks by removing or escaping special characters.
    """
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    text = text.replace('\\', '\\\\')
    text = text.replace('"', '\\"')
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = ''.join(char for char in text if char.isprintable())
    return text.strip()

def open_and_read_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Reads a PDF file and extracts text with metadata.
    """
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
        logger.error(f"Error reading PDF {pdf_path}: {str(e)}")
        raise

def split_list(input_list: List[str], slice_size: int = 10) -> List[List[str]]:
    """
    Splits a list of sentences into smaller chunks.
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

def preprocess_and_chunk(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Loads a PDF, extracts text, and creates chunks.
    """
    pages_and_texts = open_and_read_pdf(pdf_path)
    pages_and_chunks = []
    for item in pages_and_texts:
        sentence_chunks = split_list(item["sentences"], slice_size=10)
        for chunk in sentence_chunks:
            chunk_text = " ".join(chunk).strip()
            if chunk_text:
                cleaned_text = clean_text_chunk(chunk_text)
                pages_and_chunks.append({
                    "page_number": item["page_number"],
                    "sentence_chunk": cleaned_text
                })
    return pages_and_chunks

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two vectors.
    """
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def load_or_generate_embeddings(text_chunks: List[str], pdf_hash: str, 
                                progress_callback: Callable[[int, int], None] = None) -> np.ndarray:
    """
    Loads or generates embeddings for a list of text chunks.
    Accepts an optional progress_callback (unused in synchronous mode).
    """
    embeddings_path = os.path.join("embeddings", f"{pdf_hash}.npy")
    if os.path.exists(embeddings_path):
        return np.load(embeddings_path)
    
    embeddings = []
    total = len(text_chunks)
    for i in range(0, total, Config.EMBEDDING_BATCH_SIZE):
        batch = text_chunks[i:i + Config.EMBEDDING_BATCH_SIZE]
        logger.debug(f"Processing batch starting at index {i}: {batch}")
        cleaned_batch = []
        for chunk in batch:
            if not isinstance(chunk, str) or not chunk.strip():
                logger.warning(f"Skipping invalid chunk: {chunk}")
                continue
            cleaned_batch.append(clean_text_chunk(chunk))
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
                    continue
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to process batch starting at index {i}: {str(e)}")
            logger.error(f"Problematic batch: {batch}")
            continue
        if progress_callback:
            current = min(i + Config.EMBEDDING_BATCH_SIZE, total)
            progress_callback(current, total)
    if not embeddings:
        raise RuntimeError("No embeddings generated for the provided text chunks")
    embeddings_array = np.array(embeddings)
    np.save(embeddings_path, embeddings_array)
    return embeddings_array

def make_api_request(url: str, payload: dict, error_msg: str) -> dict:
    """
    Makes an API request and returns the JSON response.
    """
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"{error_msg}: {str(e)}")
        raise RuntimeError(f"{error_msg}: {str(e)}")

def ensure_complete_sentences(text: str) -> str:
    """
    Ensures the text ends with a complete sentence.
    """
    last_punctuation = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last_punctuation != -1:
        return text[:last_punctuation + 1]
    else:
        return text

def summarize_with_chat_model(text: str) -> str:
    """
    Summarizes text using the chat model.
    """
    payload = {
        "model": Config.CHAT_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an expert in text comprehension and summarization."},
            {"role": "user", "content": f"Summarize the following text in your own words:\n\n{text}"}
        ],
        "temperature": 0.6,
        "max_tokens": 500
    }
    response = requests.post(Config.LM_STUDIO_API_URL, json=payload)
    if response.status_code == 200:
        summary = response.json()["choices"][0]["message"]["content"]
        summary = ensure_complete_sentences(summary)
        return summary
    else:
        logger.error(f"Error in LLM request: {response.status_code} - {response.text}")
        return "Summary not available."

def summarize_chat_history(history: List[dict]) -> str:
    """
    Summarizes the entire chat history.
    """
    combined_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    summary_prompt = f"Summarize the following conversation by highlighting the main points:\n\n{combined_text}"
    payload = {
        "model": Config.CHAT_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an expert summarizer. Summarize the conversation in a concise manner."},
            {"role": "user", "content": summary_prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 150
    }
    try:
        response = requests.post(Config.LM_STUDIO_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        summary = response.json()["choices"][0]["message"]["content"]
        return summary
    except Exception as e:
        logger.error(f"Error summarizing chat history: {str(e)}")
        return ""

def get_llm_response(text: str) -> str:
    """
    Gets a response from the language model.
    """
    from flask import session
    chat_history = session.get("chat_history", [])
    if len(chat_history) > 10:
        summary = summarize_chat_history(chat_history)
        messages = [{"role": "system", "content": f"Conversation summary: {summary}"}]
    else:
        messages = [{"role": "system", "content": "You are a helpful assistant and your answers are quite short in one sentence to this query:"}]
        messages.extend(chat_history)
    messages.append({"role": "user", "content": text})
    payload = {
        "model": Config.CHAT_MODEL_NAME,
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": 200
    }
    response = requests.post(Config.LM_STUDIO_API_URL, json=payload)
    if response.status_code == 200:
        answer = response.json()["choices"][0]["message"]["content"]
        answer = ensure_complete_sentences(answer)
        return answer
    else:
        logger.error(f"Error in LLM request: {response.status_code} - {response.text}")
        return "Answer not available."
