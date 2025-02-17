import os
import json
import requests
import traceback
import numpy as np
import logging
from flask import Flask, render_template, request, jsonify
from sentence_transformers import util
from config import Config
from utils import (
    text_formatter,
    preprocess_and_chunk,
    load_or_generate_embeddings,
    make_api_request
)



# Initialize Flask app
app = Flask(__name__)


import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if not os.path.exists("pages_and_chunks.json"):
    pages_and_chunks = preprocess_and_chunk(Config.PDF_PATH)
    with open("pages_and_chunks.json", "w", encoding="utf-8") as f:
        json.dump(pages_and_chunks, f, indent=4)
else:
    with open("pages_and_chunks.json", "r", encoding="utf-8") as f:
        pages_and_chunks = json.load(f)

text_chunks = [chunk["sentence_chunk"] for chunk in pages_and_chunks]
embeddings = load_or_generate_embeddings(text_chunks)

# Calculate the cosine similarities
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_relevant_resources(query: str, n_resources_to_return: int = 5) -> list[tuple[dict, float]]:
    """
    Finds relevant text sections based on the query.

    Args:
        query (str): The user's query.
        n_resources_to_return (int): Number of results to return. Defaults to 5.

    Returns:
        list[tuple[dict, float]]: List of tuples containing the text chunk and its relevance score.

    Raises:
        RuntimeError: If there is an error in the API request or data processing.
    """
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

    # Convert the embedding data into a tensor
    try:
        query_embedding = np.array(query_embeddings).squeeze()
    except Exception as e:
        logger.error(f"Error converting embedding data to tensor: {e}")
        raise RuntimeError(f"Error converting embedding data to tensor: {e}")

    # Calculate the similarities
    dot_scores = np.array([cosine_similarity(query_embedding, emb) for emb in embeddings])
    indices = np.argsort(dot_scores)[-n_resources_to_return:][::-1]
    scores = dot_scores[indices]

    # Return the top results with their scores
    return [(pages_and_chunks[i], float(scores[idx])) for idx, i in enumerate(indices)]

def format_combined_summary_and_sources(results):
    """Creates a combined summary of the top 5 results and displays the sources."""
    
    # Combine the texts of the top 5 results for the combined summary
    combined_text = " ".join([text_formatter(result['sentence_chunk']) for result, _ in results])
    
    # Generate a combined summary using the chat model
    summarized_text = summarize_with_chat_model(combined_text)
    
    # Create HTML content for the original sources
    sources_html = "<strong>💡 Original Text Sources:</strong><br>"
    
    for idx, (result, score) in enumerate(results):
        original_text = text_formatter(result['sentence_chunk'])
        sources_html += (
            f"<br><strong>{idx + 1}. Score:</strong> {score:.4f}<br>"
            f"<strong>Original:</strong> {original_text}<br>"
            f"<strong>Page:</strong> {result['page_number'] + 1}<br>"
            "<hr>"
        )
    
    # Return separated contents: summary and sources
    return {
        "summary": f"<strong>📜 Summary of Top 5 Results:</strong><br>{summarized_text}",
        "sources": sources_html
    }

def get_chat_response(text):
    """Generates a response based on the relevant embeddings."""
    results = retrieve_relevant_resources(text)
    
    # Retrieve combined contents (Summary + Sources)
    formatted_content = format_combined_summary_and_sources(results)
    
    return formatted_content 

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
        print(f"Error in LLM request: {response.status_code} - {response.text}")
        return "Summary not available."

def ensure_complete_sentences(text):
    """Ensures that the text ends with a complete sentence."""
    # Find the last period, exclamation mark, or question mark
    last_punctuation = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    
    if last_punctuation != -1:
        return text[:last_punctuation + 1]  # Cuts off the text after the last sentence end
    else:
        return text  

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg")
    
    # No message provided -> raise an error
    if not msg:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        # Retrieve response data (Summary & Sources)
        response_data = get_chat_response(msg)

        return jsonify({
            "response": response_data["summary"],   # The main summary
            "sources": response_data["sources"]     # The sources as separate content
        })
    except Exception as e:
        print(f"[ERROR] Error in processing: {e}")
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")  # Show the full traceback

        return jsonify({"error": "An error occurred.", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)