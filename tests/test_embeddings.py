"""
Test cases for embedding utilities.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import shutil
import tempfile
from utils.embeddings import (
    cosine_similarity,
    make_api_request,
    load_or_generate_embeddings,
    get_all_embeddings,
    clear_embeddings_cache
)

class TestEmbeddings(unittest.TestCase):
    """Test class for embedding utility functions."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.embeddings_dir = os.path.join(self.temp_dir, "embeddings")
        self.pages_dir = os.path.join(self.temp_dir, "pages_and_chunks")
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.pages_dir, exist_ok=True)
        
        # Create test data
        self.test_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.test_pdf_hash = "test_hash"
        
        # Clear cache before each test
        clear_embeddings_cache()

    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directories
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Clear cache after each test
        clear_embeddings_cache()

    def test_cosine_similarity(self):
        """Test the cosine similarity calculation."""
        # Identical vectors
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([1, 2, 3])
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), 1.0)
        
        # Orthogonal vectors
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), 0.0)
        
        # Opposite vectors
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([-1, -2, -3])
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), -1.0)
        
        # Vector against zero vector
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([0, 0, 0])
        self.assertEqual(cosine_similarity(vec1, vec2), 0.0)
        
        # Floating point vectors
        vec1 = np.array([0.5, 0.25, 0.75])
        vec2 = np.array([0.1, 0.9, 0.2])
        self.assertTrue(0 <= cosine_similarity(vec1, vec2) <= 1)

    @patch('requests.post')
    def test_make_api_request(self, mock_post):
        """Test the API request function."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response
        
        # Test successful request
        result = make_api_request("http://test-url.com", {"test": "data"}, "Test error")
        self.assertEqual(result, {"result": "success"})
        
        # Reset mock to use the side_effect for the next call
        mock_post.reset_mock()
        
        # Test request failure with proper try/except
        mock_post.side_effect = Exception("API error")
        
        try:
            make_api_request("http://test-url.com", {"test": "data"}, "Test error")
            self.fail("Expected RuntimeError was not raised")
        except RuntimeError as e:
            # Check the error message
            self.assertIn("Test error", str(e))
            self.assertIn("API error", str(e))

    @patch('requests.post')
    def test_load_or_generate_embeddings(self, mock_post):
        """Test loading/generating embeddings."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_post.return_value = mock_response
        
        # We need to mock the file operations
        with patch('os.path.exists') as mock_exists, \
             patch('numpy.save') as mock_save, \
             patch('numpy.load') as mock_load:
                
            # Test non-existent embeddings (should generate new ones)
            mock_exists.return_value = False
            
            pdf_hash = "nonexistent_hash"
            text_chunks = ["Test sentence"]
            
            # Ensure embeddings directory exists
            os.makedirs("embeddings", exist_ok=True)
            
            embeddings = load_or_generate_embeddings(text_chunks, pdf_hash)
            self.assertIsInstance(embeddings, np.ndarray)
            
            # Test with progress callback
            progress_values = []
            def progress_callback(current, total):
                progress_values.append((current, total))
            
            load_or_generate_embeddings(text_chunks, "callback_hash", progress_callback)
            self.assertGreater(len(progress_values), 0)

    @patch('os.path.exists')
    @patch('os.path.isdir')
    @patch('os.listdir')
    @patch('numpy.load')
    @patch('json.load')
    @patch('builtins.open')
    def test_get_all_embeddings(self, mock_open, mock_json_load, mock_np_load, 
                               mock_listdir, mock_isdir, mock_exists):
        """Test loading all embeddings."""
        # Make sure the cache is clear
        clear_embeddings_cache()
        
        # Setup mocks for the first case: embeddings exist
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_listdir.return_value = ["test1.npy", "test2.npy"]
        mock_np_load.side_effect = [
            np.array([[0.1, 0.2, 0.3]]), 
            np.array([[0.4, 0.5, 0.6]])
        ]
        mock_json_load.side_effect = [
            {"pdf_name": "Test1.pdf", "chunks": [{"page_number": 0, "sentence_chunk": "Text 1"}]},
            {"pdf_name": "Test2.pdf", "chunks": [{"page_number": 0, "sentence_chunk": "Text 2"}]}
        ]
        
        # Call function for the first time
        embeddings, metadata = get_all_embeddings()
        
        # Verify results
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertIsInstance(metadata, list)
        self.assertEqual(len(metadata), 2)  # Two chunks expected
        
        # Clear cache before testing the empty case
        clear_embeddings_cache()
        
        # Reset all mocks
        mock_exists.reset_mock()
        mock_isdir.reset_mock()
        mock_listdir.reset_mock()
        mock_np_load.reset_mock()
        mock_json_load.reset_mock()
        mock_open.reset_mock()
        
        # Setup mocks for the second case: no embeddings
        mock_exists.return_value = False
        
        # Call function again
        embeddings, metadata = get_all_embeddings()
        
        # Verify empty results
        self.assertEqual(embeddings.size, 0)
        self.assertEqual(len(metadata), 0)
        
        # Test cache clearing
        clear_embeddings_cache()

if __name__ == "__main__":
    unittest.main()