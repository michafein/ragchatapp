import os
import torch
from sentence_transformers import util
import requests
from tqdm import tqdm
import fitz  # PyMuPDF
import spacy
from config import Config

# Initialize spaCy for sentence splitting
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

def text_formatter(text: str) -> str:
    """Formats the text by removing line breaks."""
    return text.replace("\n", " ").strip()

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """Reads a PDF file and extracts text with metadata."""
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
            pages_and_chunks.append({
                "page_number": item["page_number"],
                "sentence_chunk": " ".join(chunk)
            })
    return pages_and_chunks

def load_or_generate_embeddings(text_chunks, batch_size=32):
    """Loads or generates embeddings for the text chunks."""
    if os.path.exists("embeddings.pt"):
        return torch.load("embeddings.pt")
    
    embeddings = []
    for i in tqdm(range(0, len(text_chunks), batch_size)):
        batch = text_chunks[i:i + batch_size]
        payload = {"input": batch, "model": Config.EMBEDDING_MODEL_NAME}
        response = requests.post(Config.LM_STUDIO_EMBEDDING_API_URL, json=payload)
        
        if response.status_code == 200:
            embedding_data = response.json().get("data", [])
            for item in embedding_data:
                if isinstance(item, dict) and "embedding" in item:
                    embeddings.append(item["embedding"])
                else:
                    raise RuntimeError(f"Unexpected data format in API response: {item}")
        else:
            raise RuntimeError(f"Error in embedding request: {response.status_code}, {response.text}")
    
    embeddings_tensor = torch.tensor(embeddings)
    torch.save(embeddings_tensor, "embeddings.pt")
    return embeddings_tensor