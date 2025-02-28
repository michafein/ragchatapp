"""
Tests for the modularized utility functions.
"""

import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock
import numpy as np

from utils.text_processing import (
    text_formatter, 
    clean_text_chunk, 
    split_list, 
    ensure_complete_sentences
)
from utils.pdf_processing import get_pdf_hash, preprocess_and_chunk
from utils.security import sanitize_input
from utils.embeddings import cosine_similarity

class TestTextProcessing(unittest.TestCase):
    """Test cases for text processing functions."""
    
    def test_text_formatter(self):
        """Test text_formatter correctly formats text."""
        # Test newline removal
        self.assertEqual(text_formatter("Hello\nWorld"), "Hello World")
        
        # Test whitespace trimming
        self.assertEqual(text_formatter("  Hello World  "), "Hello World")
        
        # Test handling non-string input
        self.assertEqual(text_formatter(123), "123")
    
    def test_clean_text_chunk(self):
        """Test clean_text_chunk properly sanitizes text."""
        # Simple test case first
        self.assertEqual(clean_text_chunk("Simple text"), "Simple text")
        
        # Test quotes escaping
        self.assertEqual(clean_text_chunk('Text with "quotes"'), 'Text with \\"quotes\\"')
        
        # Test backslash escaping with a simpler case
        self.assertEqual(clean_text_chunk(r'Text with \backslash'), r'Text with \\backslash')
        
        # Now test the complex case that was failing
        test_input = 'Text with "quotes" and \\backslashes\\\\'
        expected = 'Text with \\"quotes\\" and \\\\backslashes\\\\\\\\'
        self.assertEqual(clean_text_chunk(test_input), expected)
    
    def test_split_list(self):
        """Test split_list divides lists correctly."""
        # Test even division
        self.assertEqual(
            split_list([1, 2, 3, 4, 5, 6], 3), 
            [[1, 2, 3], [4, 5, 6]]
        )
        
        # Test uneven division
        self.assertEqual(
            split_list([1, 2, 3, 4, 5], 2), 
            [[1, 2], [3, 4], [5]]
        )
        
        # Test empty list
        self.assertEqual(split_list([], 3), [])
    
    def test_ensure_complete_sentences(self):
        """Test ensure_complete_sentences truncates correctly."""
        # Test truncation at period
        self.assertEqual(
            ensure_complete_sentences("This is a complete sentence. This is not complete"), 
            "This is a complete sentence."
        )
        
        # Test with no punctuation
        self.assertEqual(
            ensure_complete_sentences("No punctuation here"), 
            "No punctuation here"
        )

class TestPdfProcessing(unittest.TestCase):
    """Test cases for PDF processing functions."""
    
    def test_get_pdf_hash(self):
        """Test PDF hash generation."""
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"Test PDF content")
            tmp_path = tmp.name
        
        try:
            # Test with file path
            hash1 = get_pdf_hash(tmp_path)
            self.assertIsInstance(hash1, str)
            self.assertEqual(len(hash1), 32)  # MD5 hash is 32 chars
            
            # Test with file object
            with open(tmp_path, 'rb') as f:
                hash2 = get_pdf_hash(f)
            
            # Hashes should be identical
            self.assertEqual(hash1, hash2)
        finally:
            # Clean up
            os.unlink(tmp_path)
    
    @patch('utils.pdf_processing.fitz.open')
    def test_preprocess_and_chunk(self, mock_fitz_open):
        """Test PDF preprocessing and chunking with mocks."""
        # Create fake document structure
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "This is a test sentence. Another test sentence."
        mock_doc.__iter__.return_value = [mock_page]
        mock_fitz_open.return_value = mock_doc
        
        # Setup mock sentencizer
        with patch('utils.pdf_processing.extract_sentences') as mock_extract:
            mock_extract.return_value = ["This is a test sentence.", "Another test sentence."]
            
            # Call the function with a dummy path
            result = preprocess_and_chunk("dummy_test.pdf")
            
            # Check results
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)
            self.assertIn("page_number", result[0])
            self.assertIn("sentence_chunk", result[0])

class TestSecurity(unittest.TestCase):
    """Test cases for security functions."""

    def test_sanitize_input(self):
        """Test input sanitization."""
        # Test script removal
        self.assertEqual(
            sanitize_input("Hello <script>alert('XSS')</script> World"), 
            "Hello  World"
        )
        
        # Test event handler removal - modified test to match fixed implementation
        sanitized = sanitize_input("<div onclick=\"evil()\">Click me</div>")
        # Check that the tag structure is preserved but the event handler is removed
        self.assertIn("<div", sanitized)
        self.assertIn(">Click me</div>", sanitized)
        self.assertNotIn("onclick", sanitized)
        
        # Test javascript URL removal
        self.assertEqual(
            sanitize_input("javascript:alert('XSS')"), 
            "alert('XSS')"
        )

class TestEmbeddings(unittest.TestCase):
    """Test cases for embedding functions."""
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        # Identical vectors should have similarity 1.0
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([1, 2, 3])
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), 1.0)
        
        # Orthogonal vectors should have similarity 0.0
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), 0.0)
        
        # Test with opposite vectors
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([-1, -2, -3])
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), -1.0)
        
        # Test with zero vector
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([0, 0, 0])
        self.assertEqual(cosine_similarity(vec1, vec2), 0.0)

if __name__ == "__main__":
    unittest.main()