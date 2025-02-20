import os
import json
import requests
import traceback
import numpy as np
import logging
from flask import Flask, render_template, request, jsonify, session
from config import Config
from utils import (
    text_formatter,
    preprocess_and_chunk,
    load_or_generate_embeddings,
    make_api_request,
    cosine_similarity,
    get_pdf_hash
)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'thats_my_key'

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

# Ensure directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)
os.makedirs("pages_and_chunks", exist_ok=True)

# Load or generate all embeddings
def get_all_embeddings():
    """Loads all embeddings and metadata from storage"""
    all_embeddings = []
    all_metadata = []
    
    if os.path.exists("embeddings") and os.path.isdir("embeddings"):
        for file in os.listdir("embeddings"):
            if file.endswith(".npy"):
                pdf_hash = file.split(".")[0]
                try:
                    embeddings = np.load(os.path.join("embeddings", file))
                    metadata_path = os.path.join("pages_and_chunks", f"{pdf_hash}.json")
                    
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


# if not os.path.exists("pages_and_chunks.json"):
#     pages_and_chunks = preprocess_and_chunk(Config.PDF_PATH)
#     with open("pages_and_chunks.json", "w", encoding="utf-8") as f:
#         json.dump(pages_and_chunks, f, indent=4)
# else:
#     with open("pages_and_chunks.json", "r", encoding="utf-8") as f:
#         pages_and_chunks = json.load(f)

# text_chunks = [chunk["sentence_chunk"] for chunk in pages_and_chunks]
# embeddings = load_or_generate_embeddings(text_chunks)

def retrieve_relevant_resources(query: str, n_resources_to_return: int = 5) -> list[tuple[dict, float]]:
    """
    Finds relevant text sections from all documents based on the query.

    Args:
        query (str): The user's query.
        n_resources_to_return (int): Number of results to return. Defaults to 5.

    Returns:
        list[tuple[dict, float]]: List of tuples containing the text chunk and its relevance score.

    Raises:
        RuntimeError: If there is an error in the API request or data processing.
    """
    embeddings, pages_and_chunks = get_all_embeddings()
    if embeddings.size == 0 or not pages_and_chunks:
        logger.warning("No embeddings available for search")
        return []

    # Define the payload for the embedding API request
    payload = {"input": query, "model": Config.EMBEDDING_MODEL_NAME}

    # Use the centralized API request function
    try:
        api_response = make_api_request(
            url=Config.LM_STUDIO_EMBEDDING_API_URL,
            payload=payload,
            error_msg="Embedding API Error"
        )
    except RuntimeError as e:
        logger.error(f"Failed to retrieve embeddings: {str(e)}")
        raise

    # Debugging: Log the API response
    logger.debug(f"API Response (Query): {api_response}")

    # Extract the embedding data
    embedding_data = api_response.get("data", [])
    logger.debug(f"Embedding Data (Query): {embedding_data}")

    # Validate the embedding data
    if not embedding_data or not isinstance(embedding_data, list):
        logger.error("Unexpected format of embedding data.")
        raise RuntimeError("Unexpected format of embedding data.")

    # Extract the embedding vectors
    query_embeddings = [
        item["embedding"] for item in embedding_data
        if isinstance(item, dict) and "embedding" in item
    ]
    if not query_embeddings:
        logger.error("No embedding data found in API response.")
        raise RuntimeError("No embedding data found in API response.")

    # Convert the embedding data into a numpy array
    try:
        query_embedding = np.array(query_embeddings).squeeze()
    except Exception as e:
        logger.error(f"Error converting embedding data to numpy array: {e}")
        raise RuntimeError(f"Error converting embedding data to numpy array: {e}")

    # Calculate the cosine similarities
    dot_scores = np.array([cosine_similarity(query_embedding, emb) for emb in embeddings])

    # Filter results based on the cosine similarity threshold
    above_threshold_indices = np.where(dot_scores > Config.COSINE_SIMILARITY_THRESHOLD)[0]
    above_threshold_scores = dot_scores[above_threshold_indices]

    # If no results pass the threshold, return an empty list
    if len(above_threshold_indices) == 0:
        logger.warning(f"No results found above the cosine similarity threshold of {Config.COSINE_SIMILARITY_THRESHOLD}.")
        return []

    # Sort the filtered results by score (descending) and select the top-k
    sorted_indices = above_threshold_indices[np.argsort(above_threshold_scores)[::-1]]
    sorted_scores = above_threshold_scores[np.argsort(above_threshold_scores)[::-1]]

    # Return the top results with their scores
    return [(pages_and_chunks[i], float(sorted_scores[idx])) for idx, i in enumerate(sorted_indices[:n_resources_to_return])]

def format_combined_summary_and_sources(results):
    """Creates a combined summary with PDF source information"""
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

    Args:
        text (str): The user's query.
        pdf_uploaded (bool): Whether a PDF has been uploaded. Defaults to False.

    Returns:
        dict: Contains the summary, sources, and a flag to show/hide the sources button.
    """
    if not pdf_uploaded:
        # If no PDF is uploaded, chat directly with the LLM
        summary = get_llm_response(text)
        return {
            "summary": f"<strong>📜 LLM Response:</strong><br>{summary}",
            "sources": "",
            "show_sources_button": False  
        }

    # Retrieve relevant resources if a PDF is uploaded
    results = retrieve_relevant_resources(text)

    if not results:
        # If no results are found, let the LLM respond directly
        summary = get_llm_response(text)
        return {
            "summary": f"<i> This query has no results from the PDF.</i><br><strong>📜LLM:</strong> {summary}",
            "sources": "",
            "show_sources_button": False  
        }
 

    # Format the summary and sources
    formatted_content = format_combined_summary_and_sources(results)
    return {
        "summary": formatted_content["summary"],
        "sources": formatted_content["sources"],
        "show_sources_button": True  # Show the sources button
    }

def get_llm_response(text):
    # Retrieve the existing chat history from the session
    chat_history = session.get("chat_history", [])
    
    # Decide whether to summarize the chat history if more than 10 messages exist
    if len(chat_history) > 10:
        summary = summarize_chat_history(chat_history)
        # Use the summary as the context
        messages = [{"role": "system", "content": f"Conversation summary: {summary}"}]
    else:
        # Otherwise, use the full chat history
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages.extend(chat_history)
    
    # Append the current user query
    messages.append({"role": "user", "content": text})
    
    payload = {
        "model": Config.CHAT_MODEL_NAME,
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": 500
    }
    
    response = requests.post(Config.LM_STUDIO_API_URL, json=payload)
    
    if response.status_code == 200:
        answer = response.json()["choices"][0]["message"]["content"]
        answer = ensure_complete_sentences(answer)
        return answer
    else:
        logger.error(f"Error in LLM request: {response.status_code} - {response.text}")
        return "Answer not available."


def summarize_with_chat_model(text):
    """Uses the chat model to generate a summary of the text in its own words."""
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
        # Post-processing: removes incomplete sentences at the end
        summary = ensure_complete_sentences(summary)
        return summary
    else:
        logger.error(f"Error in LLM request: {response.status_code} - {response.text}")
        return "Summary not available."

def ensure_complete_sentences(text):
    """Ensures that the text ends with a complete sentence."""
    # Find the last period, exclamation mark, or question mark
    last_punctuation = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    
    if last_punctuation != -1:
        return text[:last_punctuation + 1]  # Cuts off the text after the last sentence end
    else:
        return text  
    
def summarize_chat_history(history: list[dict]) -> str:
    """
    Summarizes the entire chat history.
    
    Args:
        history (list[dict]): A list of messages (e.g., {"role": "user"/"assistant", "content": ...})
        
    Returns:
        str: A concise summary of the conversation.
    """
    # Combine all messages into a single text
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



@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg")
    
    # No message provided -> raise an error
    if not msg:
        return jsonify({"error": "No message provided"}), 400
    
    # Initialize chat history in session if not exists
    if "chat_history" not in session:
        session["chat_history"] = []

    # Add user message to history
    session["chat_history"].append({"role": "user", "content": msg})
    
    try:
        # Check if a PDF has been uploaded
        pdf_uploaded = any(os.listdir("embeddings")) 

        # Retrieve response with context
        response_data = get_chat_response(msg, pdf_uploaded=pdf_uploaded)

        # Add assistant response to history
        session["chat_history"].append({"role": "assistant", "content": response_data["summary"]})

        return jsonify({
            "response": response_data["summary"],   # The main summary
            "sources": response_data["sources"],    # The sources as separate content
            "show_sources_button": response_data["show_sources_button"]  # Whether to show the sources button
        })
    except Exception as e:
        logger.error(f"Error in processing: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")  # Show the full traceback

        return jsonify({"error": "An error occurred.", "details": str(e)}), 500



@app.route('/upload', methods=["POST"])
def upload_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if not file.filename.endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400
            
        # Generate hash from file stream
        pdf_hash = get_pdf_hash(file.stream)
        
        # Save file to uploads directory
        file_path = os.path.join("uploads", f"{pdf_hash}.pdf")
        file.save(file_path)

        # Process PDF
        pages_and_chunks = preprocess_and_chunk(file_path)
        
        # Save metadata with PDF name
        metadata = {
            "pdf_name": file.filename,
            "chunks": pages_and_chunks
        }
        metadata_path = os.path.join("pages_and_chunks", f"{pdf_hash}.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        
        # Generate embeddings
        text_chunks = [chunk["sentence_chunk"] for chunk in pages_and_chunks]
        load_or_generate_embeddings(text_chunks, pdf_hash)
        
        return jsonify({
            "message": "PDF processed successfully",
            "pdf_name": file.filename
        }), 200
    
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return jsonify({
            "error": "Failed to process PDF",
            "details": str(e)
        }), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)