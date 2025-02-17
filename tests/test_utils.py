import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import fitz  # PyMuPDF
from utils import (
    text_formatter,
    open_and_read_pdf,
    split_list,
    preprocess_and_chunk,
    load_or_generate_embeddings,
    nlp
)

class TestUtils(unittest.TestCase):
    def test_text_formatter(self):
        """Testet die text_formatter-Funktion."""
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
        """Testet die open_and_read_pdf-Funktion."""
        # Create a small test PDF file
        pdf_path = "test.pdf"
        with fitz.open() as doc:
            page = doc.new_page()
            page.insert_text((50, 50), "This is a test sentence.")
            doc.save(pdf_path)

        # Test if the function extracts text correctly
        result = open_and_read_pdf(pdf_path)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(item, dict) for item in result))
        self.assertTrue(all("page_number" in item and "text" in item and "sentences" in item for item in result))

         # Cleanup: Delete the test PDF file
        os.remove(pdf_path)

    def test_split_list(self):
        """Testet die split_list-Funktion."""
         # Test if the list is split correctly
        input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expected_output = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        self.assertEqual(split_list(input_list, slice_size=5), expected_output)

        # Test if the remainder is handled correctly
        input_list = [1, 2, 3, 4, 5, 6]
        expected_output = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(split_list(input_list, slice_size=3), expected_output)

        # Test if an empty list is handled correctly 
        input_list = []
        expected_output = []
        self.assertEqual(split_list(input_list, slice_size=3), expected_output)

    def test_preprocess_and_chunk(self):
        """Testet die preprocess_and_chunk-Funktion."""
        # Create a small test PDF file 
        pdf_path = "test.pdf"
        with fitz.open() as doc:
            page = doc.new_page()
            page.insert_text((50, 50), "This is a test sentence. Another one.")
            doc.save(pdf_path)

        # Test if the function creates text chunks correctly
        result = preprocess_and_chunk(pdf_path)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(item, dict) for item in result))
        self.assertTrue(all("page_number" in item and "sentence_chunk" in item for item in result))

        # Cleanup: Delete the test PDF file 
        os.remove(pdf_path)

    @patch("requests.post")
    def test_load_or_generate_embeddings(self, mock_post):
        """Testet die load_or_generate_embeddings-Funktion."""
        # Mock the API response  
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_post.return_value = mock_response

        # Test if embeddings are correctly generated  
        text_chunks = ["Test sentence"]
        embeddings = load_or_generate_embeddings(text_chunks)
        self.assertIsInstance(embeddings, np.ndarray)  

        # Test if existing embeddings are loaded
        with patch("os.path.exists", return_value=True), patch("numpy.load", return_value=np.array([[0.1, 0.2, 0.3]])):
            embeddings = load_or_generate_embeddings(text_chunks)
            self.assertIsInstance(embeddings, np.ndarray)  

    def test_nlp_pipeline(self):
        """Testet die spaCy-Pipeline."""
        # Test if the pipeline is correctly initialized  
        text = "This is a test sentence. Another one."
        doc = nlp(text)
        self.assertEqual(len(list(doc.sents)), 2)  # There should be two sentences detected 

if __name__ == "__main__":
    unittest.main()