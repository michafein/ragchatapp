"""
Utility modules for the RAG chatbot application.
This file provides backward compatibility with the original monolithic structure.
"""

import logging

# Import from sub-modules
from utils.text_processing import (
    text_formatter,
    clean_text_chunk,
    split_list,
    ensure_complete_sentences,
    extract_sentences,
    nlp  # Important for backward compatibility
)

from utils.pdf_processing import (
    get_pdf_hash,
    open_and_read_pdf,
    preprocess_and_chunk,
)

from utils.embeddings import (
    cosine_similarity,
    make_api_request,
    load_or_generate_embeddings,
    get_all_embeddings,
    clear_embeddings_cache
)

from utils.security import (
    validate_file_upload,
    sanitize_input,
    set_secure_headers,
    validate_api_request
)

# Set up logging
logger = logging.getLogger(__name__)

# For complete backward compatibility, redefine functions here
# that might be called directly from routes.py

# Function declarations for backward compatibility
# This is only needed if your routes.py accesses these directly from utils
# and not through their modules

# Example:
# def format_combined_summary_and_sources(results):
#     """
#     Creates a combined summary with PDF source information.
#     Redirects to the function in the appropriate module.
#     """
#     # Import here to avoid circular imports
#     from routes import format_combined_summary_and_sources as actual_fn
#     return actual_fn(results)

# Include any other functions that were directly in utils.py and
# accessed by other modules