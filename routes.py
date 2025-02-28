"""
Route definitions for the RAG Chatbot application.
"""

import os
import json
import logging
import traceback
import numpy as np
from flask import Blueprint, render_template, request, jsonify, session, Response
import requests
import queue
import threading
import fitz  # PyMuPDF
import time

from config import Config
from utils import (
    text_formatter,
    preprocess_and_chunk,
    load_or_generate_embeddings,
    make_api_request,
    cosine_similarity,
    get_pdf_hash,
    ensure_complete_sentences,
    sanitize_input
)

# Initialize Blueprint and logger
main_bp = Blueprint('main', __name__)
logger = logging.getLogger(__name__)

@main_bp.route('/')
def index():
    """Render the main chat interface."""
    return render_template('chat.html')

@main_bp.route('/get', methods=["GET", "POST"])
def chat():
    """Handle chat requests from the user."""
    msg = request.form.get("msg")
    if not msg:
        return jsonify({"error": "No message provided"}), 400

    # Sanitize input
    msg = sanitize_input(msg)

    # Initialize chat history if needed
    if "chat_history" not in session:
        session["chat_history"] = []
    session["chat_history"].append({"role": "user", "content": msg})

    try:
        # Check if any PDFs have been uploaded
        pdf_uploaded = any(os.path.isdir("embeddings") and os.listdir("embeddings"))
        
        # Get chat response
        response_data = get_chat_response(msg, pdf_uploaded=pdf_uploaded)
        
        # Add assistant response to history
        session["chat_history"].append({
            "role": "assistant", 
            "content": response_data["summary"]
        })
        
        # Return the response
        return jsonify({
            "response": response_data["summary"],
            "sources": response_data["sources"],
            "show_sources_button": response_data["show_sources_button"]
        })
    
    except Exception as e:
        logger.error(f"Error in processing: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": "An error occurred.", "details": str(e)}), 500

def get_all_embeddings():
    """
    Loads all embeddings and metadata from storage.
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
                        with open(metadata_path, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                            # Add PDF name to each chunk
                            for chunk in metadata["chunks"]:
                                chunk["pdf_name"] = metadata["pdf_name"]
                            all_metadata.extend(metadata["chunks"])
                            all_embeddings.append(embeddings)
                except Exception as e:
                    logger.error(f"Error processing {file}: {str(e)}")
                    continue
    if not all_embeddings:
        return np.array([]), []
    return np.concatenate(all_embeddings), all_metadata

def retrieve_relevant_resources(query: str, n_resources_to_return: int = 5) -> list:
    """
    Finds relevant text sections from all documents based on the query.
    """
    embeddings, pages_and_chunks = get_all_embeddings()
    if embeddings.size == 0 or not pages_and_chunks:
        logger.warning("No embeddings available for search")
        return []

    payload = {"input": query, "model": Config.EMBEDDING_MODEL_NAME}
    try:
        api_response = make_api_request(
            url=Config.LM_STUDIO_EMBEDDING_API_URL,
            payload=payload,
            error_msg="Embedding API Error"
        )
    except RuntimeError as e:
        logger.error(f"Failed to retrieve embeddings: {str(e)}")
        raise

    logger.debug(f"API Response (Query): {api_response}")
    embedding_data = api_response.get("data", [])
    logger.debug(f"Embedding Data (Query): {embedding_data}")

    if not embedding_data or not isinstance(embedding_data, list):
        logger.error("Unexpected format of embedding data.")
        raise RuntimeError("Unexpected format of embedding data.")

    query_embeddings = [
        item["embedding"] for item in embedding_data
        if isinstance(item, dict) and "embedding" in item
    ]
    if not query_embeddings:
        logger.error("No embedding data found in API response.")
        raise RuntimeError("No embedding data found in API response.")

    try:
        query_embedding = np.array(query_embeddings).squeeze()
    except Exception as e:
        logger.error(f"Error converting embedding data to numpy array: {e}")
        raise RuntimeError(f"Error converting embedding data to numpy array: {e}")

    dot_scores = np.array([cosine_similarity(query_embedding, emb) for emb in embeddings])
    above_threshold_indices = np.where(dot_scores > Config.COSINE_SIMILARITY_THRESHOLD)[0]
    above_threshold_scores = dot_scores[above_threshold_indices]

    if len(above_threshold_indices) == 0:
        logger.warning(f"No results found above the cosine similarity threshold of {Config.COSINE_SIMILARITY_THRESHOLD}.")
        return []

    sorted_order = np.argsort(above_threshold_scores)[::-1]
    sorted_indices = above_threshold_indices[sorted_order]
    sorted_scores = above_threshold_scores[sorted_order]

    return [(pages_and_chunks[i], float(sorted_scores[idx])) for idx, i in enumerate(sorted_indices[:n_resources_to_return])]

def format_combined_summary_and_sources(results):
    """
    Creates a combined summary with PDF source information.
    """
    if not results:
        return {
            "summary": "No relevant information found.",
            "sources": ""
        }
    combined_text = " ".join([text_formatter(result['sentence_chunk']) for result, _ in results])
    summarized_text = summarize_with_chat_model(combined_text)
    sources_html = "<strong>💡 Original Text Sources:</strong><br>"
    for idx, (result, score) in enumerate(results):
        sources_html += (
            f"<br><strong>{idx + 1}. Source:</strong> {result.get('pdf_name', 'Unknown document')}<br>"
            f"<strong>Score:</strong> {score:.4f}<br>"
            f"<strong>Text:</strong> {text_formatter(result['sentence_chunk'])}<br>"
            f"<strong>Page:</strong> {result['page_number'] + 1}<br>"
            "<hr>"
        )
    return {
        "summary": f"<strong>📜 Summary of Top Results:</strong><br>{summarized_text}",
        "sources": sources_html
    }

def get_chat_response(text, pdf_uploaded: bool = False):
    """
    Generates a response based on the relevant embeddings.
    """
    if not pdf_uploaded:
        summary = get_llm_response(text)
        return {
            "summary": f"<strong>📜 LLM Response:</strong><br>{summary}",
            "sources": "",
            "show_sources_button": False
        }

    results = retrieve_relevant_resources(text)
    if not results:
        summary = get_llm_response(text)
        return {
            "summary": f"<i>This query has no results from the PDF.</i><br><strong>📜LLM:</strong> {summary}",
            "sources": "",
            "show_sources_button": False
        }

    formatted_content = format_combined_summary_and_sources(results)
    return {
        "summary": formatted_content["summary"],
        "sources": formatted_content["sources"],
        "show_sources_button": True
    }


@main_bp.route('/upload_stream', methods=["POST"])
def upload_pdf_stream():
    """
    Upload and process a PDF file with streaming progress updates.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    # Read file content once
    file_content = file.read()
    
    # Ensure directories exist
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)
    os.makedirs("pages_and_chunks", exist_ok=True)

    # Generate hash immediately to check for existing embeddings
    try:
        pdf_hash = get_pdf_hash(file_content)
        embeddings_path = os.path.join("embeddings", f"{pdf_hash}.npy")
        
        # Check if embeddings already exist
        if os.path.exists(embeddings_path):
            # Return a special message for duplicates
            return Response(
                "data: PDF already processed\n\n"
                f"data: Embeddings for '{file.filename}' already exist\n\n"
                "data: Status: duplicate\n\n",
                mimetype="text/event-stream"
            )
    except Exception as e:
        logger.error(f"Error generating hash: {e}")
        return Response(
            f"data: Error: Failed to process PDF: {str(e)}\n\n",
            mimetype="text/event-stream"
        )

    # Create a queue for progress updates
    progress_queue = queue.Queue()

    def process_pdf():
        """Process PDF and generate embeddings in a separate thread"""
        try:
            # We already computed the hash above
            file_path = os.path.join("uploads", f"{pdf_hash}.pdf")
            
            # Save PDF file
            with open(file_path, "wb") as f:
                f.write(file_content)
            progress_queue.put(f"data: PDF saved with hash {pdf_hash}\n\n")
            progress_queue.put("data: Progress: 10%\n\n")

            # Verify PDF is valid and extract text
            try:
                # Try to open the PDF to see if it's valid
                try:
                    pdf_document = fitz.open(file_path)
                    if pdf_document.needs_pass:
                        pdf_document.close()
                        raise ValueError("This PDF is password-protected. Please remove the password protection and try again.")
                    
                    # Check if PDF has content
                    if pdf_document.page_count == 0:
                        pdf_document.close()
                        raise ValueError("The PDF contains no pages.")
                    
                    # Check if any page has text
                    has_text = False
                    for page in pdf_document:
                        if page.get_text().strip():
                            has_text = True
                            break
                    
                    if not has_text:
                        pdf_document.close()
                        raise ValueError("The PDF does not contain any extractable text. It may contain only images or scanned content without OCR.")
                    
                    pdf_document.close()
                except fitz.FileDataError:
                    raise ValueError("The file appears to be corrupted or not a valid PDF.")
                
                # Process PDF to extract content
                pages_and_chunks = preprocess_and_chunk(file_path)
                if not pages_and_chunks:
                    raise ValueError("No text content could be extracted from this PDF. The file may be empty or contain only non-textual elements.")
                
                progress_queue.put(f"data: PDF processed, {len(pages_and_chunks)} text chunks extracted\n\n")
                progress_queue.put("data: Progress: 20%\n\n")
            except Exception as e:
                error_msg = str(e)
                logger.error(f"PDF processing error: {error_msg}")
                progress_queue.put(f"data: Error: {error_msg}\n\n")
                progress_queue.put("data: Status: error\n\n")
                progress_queue.put(None)  # End processing
                return

            # Save metadata
            try:
                metadata = {"pdf_name": file.filename, "chunks": pages_and_chunks}
                metadata_path = os.path.join("pages_and_chunks", f"{pdf_hash}.json")
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=4)
                progress_queue.put("data: Metadata saved\n\n")
                progress_queue.put("data: Progress: 30%\n\n")
            except Exception as e:
                logger.error(f"Metadata saving error: {e}")
                progress_queue.put(f"data: Error: Failed to save document metadata: {str(e)}\n\n")
                progress_queue.put("data: Status: error\n\n")
                progress_queue.put(None)
                return

            # Generate embeddings in batches
            try:
                text_chunks = [chunk["sentence_chunk"] for chunk in pages_and_chunks]
                if not text_chunks:
                    raise ValueError("No valid text chunks were extracted from the PDF.")
                
                # Process embeddings in smaller batches
                total_chunks = len(text_chunks)
                batch_size = Config.EMBEDDING_BATCH_SIZE
                
                all_embeddings = []
                
                for i in range(0, total_chunks, batch_size):
                    batch = text_chunks[i:i + batch_size]
                    
                    # Process this batch
                    payload = {
                        "input": batch,
                        "model": Config.EMBEDDING_MODEL_NAME
                    }
                    
                    try:
                        response = requests.post(
                            Config.LM_STUDIO_EMBEDDING_API_URL,
                            json=payload,
                            timeout=120
                        )
                        response.raise_for_status()
                        
                        # Extract embedding data
                        embedding_data = response.json().get("data", [])
                        
                        batch_embeddings = []
                        for item in embedding_data:
                            if isinstance(item, dict) and "embedding" in item:
                                batch_embeddings.append(item["embedding"])
                        
                        all_embeddings.extend(batch_embeddings)
                    except requests.exceptions.RequestException as e:
                        connection_error = "Connection refused" in str(e) or "Failed to establish a connection" in str(e)
                        if connection_error:
                            error_msg = "Could not connect to the embedding API. Please ensure LM-Studio is running."
                        else:
                            error_msg = f"Failed to process embeddings batch: {str(e)}"
                        
                        logger.error(f"API error: {error_msg}")
                        progress_queue.put(f"data: Error: {error_msg}\n\n")
                        progress_queue.put("data: Status: error\n\n")
                        progress_queue.put(None)
                        return
                    
                    # Calculate and report progress - scale from 30% to 90%
                    current = min(i + batch_size, total_chunks)
                    progress_percent = 30 + int((current / total_chunks) * 60)
                    progress_queue.put(f"data: Progress: {progress_percent}%\n\n")
                
                # Check if we got any embeddings
                if not all_embeddings:
                    raise ValueError("Failed to generate embeddings for this PDF. The text may not be suitable for embedding.")
                
                # Save all embeddings
                embeddings_array = np.array(all_embeddings)
                np.save(embeddings_path, embeddings_array)
                
                # Send 100% progress update first
                progress_queue.put("data: Progress: 100%\n\n")
                
                # Short delay to ensure progress bar reaches 100% before completion message
                time.sleep(0.5)
                
                # Then send completion message and final status
                progress_queue.put("data: Embeddings generated and saved. Process complete.\n\n")
                progress_queue.put("data: Status: success\n\n")
            except Exception as e:
                logger.error(f"Embedding generation error: {e}")
                progress_queue.put(f"data: Error: {str(e)}\n\n")
                progress_queue.put("data: Status: error\n\n")
                progress_queue.put(None)
                return
                
        except Exception as e:
            logger.error(f"Error during PDF processing: {e}")
            logger.error(traceback.format_exc())
            progress_queue.put(f"data: Error: {str(e)}\n\n")
            progress_queue.put("data: Status: error\n\n")
        
        # Mark the end of processing
        progress_queue.put(None)

    def generate():
        # Start the processing thread
        thread = threading.Thread(target=process_pdf)
        thread.daemon = True  # Thread will be killed when the main thread exits
        thread.start()
        
        # Yield updates as they become available
        while True:
            try:
                update = progress_queue.get(timeout=180.0)  # Wait up to 3 minutes for updates
                if update is None:  # End marker
                    break
                yield update
            except queue.Empty:
                # No update received within timeout
                yield "data: Error: Timed out waiting for updates. The process might still be running, but no progress has been reported.\n\n"
                yield "data: Status: error\n\n"
                break

    return Response(generate(), mimetype="text/event-stream")

@main_bp.route('/upload', methods=["POST"])
def upload_pdf():
    """
    Synchronous endpoint to upload and process a PDF.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({"error": "Only PDF files are allowed"}), 400
    
    # Ensure directories exist
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)
    os.makedirs("pages_and_chunks", exist_ok=True)
    
    try:
        # Read file content
        file_content = file.read()
        
        # Get PDF hash
        pdf_hash = get_pdf_hash(file_content)
        
        # Save PDF
        file_path = os.path.join("uploads", f"{pdf_hash}.pdf")
        with open(file_path, "wb") as f:
            f.write(file_content)
            
        # Process PDF
        pages_and_chunks = preprocess_and_chunk(file_path)
        
        # Save metadata
        metadata = {
            "pdf_name": file.filename,
            "chunks": pages_and_chunks
        }
        metadata_path = os.path.join("pages_and_chunks", f"{pdf_hash}.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
            
        # Generate and save embeddings
        text_chunks = [chunk["sentence_chunk"] for chunk in pages_and_chunks]
        
        # This call should both generate and save the embeddings
        embeddings = load_or_generate_embeddings(text_chunks, pdf_hash)
        
        # Verify embeddings were saved
        embeddings_path = os.path.join("embeddings", f"{pdf_hash}.npy")
        if not os.path.exists(embeddings_path):
            logger.warning(f"Expected embeddings file not found at {embeddings_path}")
        
        return jsonify({
            "message": "PDF processed successfully.",
            "pdf_hash": pdf_hash,
            "pdf_name": file.filename,
            "embeddings_saved": os.path.exists(embeddings_path)
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "PDF processing failed", "details": str(e)}), 500

@main_bp.route('/clear_history', methods=["POST"])
def clear_history():
    """
    Clear the chat history from the session.
    """
    session.pop("chat_history", None)
    return jsonify({"message": "Chat history cleared"}), 200

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

def summarize_chat_history(history: list) -> str:
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