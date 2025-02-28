"""
Test cases for utility functions used in the RAG chatbot application.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import fitz  # PyMuPDF

# Import from the new modular structure
from utils.text_processing import text_formatter, nlp, extract_sentences
from utils.pdf_processing import open_and_read_pdf, preprocess_and_chunk
from utils.embeddings import load_or_generate_embeddings

class TestUtils(unittest.TestCase):
    """Test class for utility functions."""
    
    def test_text_formatter(self):
        """Test the text_formatter function."""
        # Test if line breaks are removed
        input_text = "Hello\nWorld"
        expected_output = "Hello World"
        self.assertEqual(text_formatter(input_text), expected_output)

        # Test if leading/trailing spaces are removed
        input_text = "  Hello World  "
        expected_output = "Hello World"
        self.assertEqual(text_formatter(input_text), expected_output)

        # Test if multiple line breaks are handled correctly
        input_text = "Line1\nLine2\nLine3"
        expected_output = "Line1 Line2 Line3"
        self.assertEqual(text_formatter(input_text), expected_output)

    def test_open_and_read_pdf(self):
        """Test the open_and_read_pdf function."""
        # Create a small test PDF file
        pdf_path = "test.pdf"
        with fitz.open() as doc:
            page = doc.new_page()
            page.insert_text((50, 50), "This is a test sentence.")
            doc.save(pdf_path)

        try:
            # Test if the function extracts text correctly
            result = open_and_read_pdf(pdf_path)
            self.assertIsInstance(result, list)
            self.assertTrue(all(isinstance(item, dict) for item in result))
            self.assertTrue(all("page_number" in item and "text" in item and "sentences" in item for item in result))
        finally:
            # Cleanup: Delete the test PDF file
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    def test_preprocess_and_chunk(self):
        """Test the preprocess_and_chunk function."""
        # Create a small test PDF file 
        pdf_path = "test.pdf"
        with fitz.open() as doc:
            page = doc.new_page()
            page.insert_text((50, 50), "This is a test sentence. Another one.")
            doc.save(pdf_path)

        try:
            # Test if the function creates text chunks correctly
            result = preprocess_and_chunk(pdf_path)
            self.assertIsInstance(result, list)
            self.assertTrue(all(isinstance(item, dict) for item in result))
            self.assertTrue(all("page_number" in item and "sentence_chunk" in item for item in result))
        finally:
            # Cleanup: Delete the test PDF file 
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    @patch("requests.post")
    def test_load_or_generate_embeddings(self, mock_post):
        """Test the load_or_generate_embeddings function."""
        # Mock the API response  
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_post.return_value = mock_response

        # Use a temporary PDF hash for testing
        pdf_hash = "test_hash"
        
        # Ensure the embeddings directory exists
        os.makedirs("embeddings", exist_ok=True)
        
        try:
            # Test if embeddings are correctly generated  
            text_chunks = ["Test sentence"]
            embeddings = load_or_generate_embeddings(text_chunks, pdf_hash)
            self.assertIsInstance(embeddings, np.ndarray)
        finally:
            # Clean up any generated files
            if os.path.exists(f"embeddings/{pdf_hash}.npy"):
                os.remove(f"embeddings/{pdf_hash}.npy")

    def test_nlp_pipeline(self):
        """Test the spaCy NLP pipeline."""
        # Test if the pipeline is correctly initialized  
        text = "This is a test sentence. Another one."
        sentences = extract_sentences(text)
        self.assertEqual(len(sentences), 2)  # There should be two sentences detected

if __name__ == "__main__":
    unittest.main()