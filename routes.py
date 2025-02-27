import os
import json
import logging
import traceback
import numpy as np
from flask import Blueprint, render_template, request, jsonify, session, Response
import requests
from config import Config
from utils import (
    text_formatter,
    preprocess_and_chunk,
    load_or_generate_embeddings,
    make_api_request,
    cosine_similarity,
    get_pdf_hash,
    ensure_complete_sentences,
    summarize_with_chat_model,
    summarize_chat_history,
    get_llm_response
)

main_bp = Blueprint('main', __name__)
logger = logging.getLogger(__name__)

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

@main_bp.route('/')
def index():
    return render_template('chat.html')

@main_bp.route('/get', methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg")
    if not msg:
        return jsonify({"error": "No message provided"}), 400

    if "chat_history" not in session:
        session["chat_history"] = []
    session["chat_history"].append({"role": "user", "content": msg})

    try:
        pdf_uploaded = any(os.listdir("embeddings"))
        response_data = get_chat_response(msg, pdf_uploaded=pdf_uploaded)
        session["chat_history"].append({"role": "assistant", "content": response_data["summary"]})
        return jsonify({
            "response": response_data["summary"],
            "sources": response_data["sources"],
            "show_sources_button": response_data["show_sources_button"]
        })
    except Exception as e:
        logger.error(f"Error in processing: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": "An error occurred.", "details": str(e)}), 500

@main_bp.route('/upload_stream', methods=["POST"])
def upload_pdf_stream():
    """
    Synchronous endpoint to upload and process a PDF, providing progress updates via Server-Sent Events.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    # Read file content once outside of the generator to avoid later access to a closed file.
    file_content = file.read()

    def generate():
        try:
            import hashlib
            # Compute hash directly from the file content
            pdf_hash = hashlib.md5(file_content).hexdigest()
            file_path = os.path.join("uploads", f"{pdf_hash}.pdf")
            with open(file_path, "wb") as f:
                f.write(file_content)
            yield f"data: PDF saved with hash {pdf_hash}\n\n"

            # Process PDF: extract text and create chunks
            pages_and_chunks = preprocess_and_chunk(file_path)
            yield f"data: PDF processed, {len(pages_and_chunks)} text chunks extracted\n\n"

            # Save metadata
            metadata = {"pdf_name": file.filename, "chunks": pages_and_chunks}
            metadata_path = os.path.join("pages_and_chunks", f"{pdf_hash}.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            yield "data: Metadata saved\n\n"

            # Generate embeddings with progress updates
            text_chunks = [chunk["sentence_chunk"] for chunk in pages_and_chunks]
            total = len(text_chunks)
            embeddings = []
            for i in range(0, total, Config.EMBEDDING_BATCH_SIZE):
                batch = text_chunks[i:i + Config.EMBEDDING_BATCH_SIZE]
                cleaned_batch = [chunk.strip() for chunk in batch if chunk.strip()]
                if not cleaned_batch:
                    continue
                payload = {
                    "input": cleaned_batch,
                    "model": Config.EMBEDDING_MODEL_NAME
                }
                response = requests.post(Config.LM_STUDIO_EMBEDDING_API_URL, json=payload, timeout=120)
                response.raise_for_status()
                embedding_data = response.json().get("data", [])
                for item in embedding_data:
                    if isinstance(item, dict) and "embedding" in item:
                        embeddings.append(item["embedding"])
                current = min(i + Config.EMBEDDING_BATCH_SIZE, total)
                progress_percent = int((current / total) * 100)
                yield f"data: Progress: {progress_percent}%\n\n"
            embeddings_array = np.array(embeddings)
            np.save(os.path.join("embeddings", f"{pdf_hash}.npy"), embeddings_array)
            yield "data: Embeddings generated and saved. Process complete.\n\n"
        except Exception as e:
            logger.error(f"Error during PDF processing: {e}")
            yield f"data: Error: {str(e)}\n\n"

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
    try:
        # Read file content as bytes
        file_content = file.read()
        from io import BytesIO
        file_stream = BytesIO(file_content)
        pdf_hash = get_pdf_hash(file_stream)
        # Save PDF to uploads folder
        file_path = os.path.join("uploads", f"{pdf_hash}.pdf")
        with open(file_path, "wb") as f:
            f.write(file_content)
        # Process PDF to extract text chunks
        pages_and_chunks = preprocess_and_chunk(file_path)
        # Save metadata (PDF name and chunks)
        metadata = {
            "pdf_name": file.filename,
            "chunks": pages_and_chunks
        }
        metadata_path = os.path.join("pages_and_chunks", f"{pdf_hash}.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        # Generate embeddings synchronously for all text chunks
        text_chunks = [chunk["sentence_chunk"] for chunk in pages_and_chunks]
        load_or_generate_embeddings(text_chunks, pdf_hash)
        return jsonify({
            "message": "PDF processed successfully.",
            "pdf_hash": pdf_hash,
            "pdf_name": file.filename
        }), 200
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return jsonify({"error": "PDF processing failed", "details": str(e)}), 500
