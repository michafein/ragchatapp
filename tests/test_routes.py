"""
Test cases for route handlers and API endpoints.
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import io
import os
import sys
import numpy as np
import tempfile
from flask import session

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app

class TestRoutes(unittest.TestCase):
    """Test class for route handlers and helper functions."""

    def setUp(self):
        """Set up test environment."""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        """Clean up after tests."""
        self.app_context.pop()

    def test_index_route(self):
        """Test the index route."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<!DOCTYPE html>', response.data)

    def test_chat_route(self):
        """Test the chat endpoint."""
        # Test with empty message
        response = self.client.post('/get', data={})
        self.assertEqual(response.status_code, 400)
        
        # Test case 1: No PDFs uploaded - Mock os.listdir to return empty list
        with patch('os.listdir', return_value=[]) as mock_listdir:
            with patch('requests.post') as mock_post:
                # Mock the response from the LLM API
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Test response"}}]
                }
                mock_post.return_value = mock_response
                
                # Test with valid message
                response = self.client.post('/get', data={"msg": "test message"})
                self.assertEqual(response.status_code, 200)
                
                # Verify response structure
                data = json.loads(response.data)
                self.assertIn("response", data)
        
        # Test case 2: PDFs uploaded - Mock both the PDF check and retrieve_relevant_resources
        with patch('os.listdir', return_value=["test.npy"]) as mock_listdir:
            with patch('routes.retrieve_relevant_resources', return_value=[]) as mock_retrieve:
                with patch('requests.post') as mock_post:
                    # Mock the response from the LLM API
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "choices": [{"message": {"content": "Test response"}}]
                    }
                    mock_post.return_value = mock_response
                    
                    # Test with valid message
                    response = self.client.post('/get', data={"msg": "test with pdf"})
                    self.assertEqual(response.status_code, 200)
                    
                    # Verify response structure
                    data = json.loads(response.data)
                    self.assertIn("response", data)
        
        # Test error handling by forcing an exception
        with patch('os.listdir', return_value=[]) as mock_listdir:
            with patch('requests.post') as mock_post:
                mock_post.side_effect = Exception("Test error")
                response = self.client.post('/get', data={"msg": "error message"})
                self.assertEqual(response.status_code, 500)
                data = json.loads(response.data)
                self.assertIn("error", data)

    def test_clear_history_route(self):
        """Test the clear history endpoint."""
        # Add data to session
        with self.client.session_transaction() as sess:
            sess["chat_history"] = [{"role": "user", "content": "test"}]
        
        # Call clear_history
        response = self.client.post('/clear_history')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["message"], "Chat history cleared")
        
        # Check that session was cleared
        with self.client.session_transaction() as sess:
            self.assertNotIn("chat_history", sess)

    # Completely reworked this test to work with BytesIO and proper mocking
    def test_upload_pdf(self):
        """Test the PDF upload endpoint."""
        # Create a temporary PDF in memory
        pdf_data = b"%PDF-1.5\nTest PDF content"
        
        # Mock the entire route internals
        with patch('routes.get_pdf_hash') as mock_hash, \
             patch('routes.preprocess_and_chunk') as mock_chunk, \
             patch('routes.load_or_generate_embeddings') as mock_embed, \
             patch('builtins.open', create=True), \
             patch('json.dump'):
            
            # Setup mocks
            mock_hash.return_value = "test_hash"
            mock_chunk.return_value = [{"page_number": 0, "sentence_chunk": "Test sentence"}]
            mock_embed.return_value = np.array([[0.1, 0.2, 0.3]])
            
            # Create directories
            os.makedirs("uploads", exist_ok=True)
            os.makedirs("pages_and_chunks", exist_ok=True)
            os.makedirs("embeddings", exist_ok=True)
            
            # Test with valid PDF
            response = self.client.post(
                '/upload',
                data={'file': (io.BytesIO(pdf_data), 'test.pdf')},
                content_type='multipart/form-data'
            )
            
            # Should be successful
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('message', data)
            self.assertEqual(data['message'], 'PDF processed successfully.')
            
            # Test with no file
            response = self.client.post('/upload')
            self.assertEqual(response.status_code, 400)
            
            # Test with non-PDF file
            response = self.client.post(
                '/upload',
                data={'file': (io.BytesIO(b"Not a PDF"), 'test.txt')},
                content_type='multipart/form-data'
            )
            self.assertEqual(response.status_code, 400)
            
            # Test with error in processing
            mock_chunk.side_effect = Exception("Test error")
            response = self.client.post(
                '/upload',
                data={'file': (io.BytesIO(pdf_data), 'error.pdf')},
                content_type='multipart/form-data'
            )
            self.assertEqual(response.status_code, 500)

    def test_upload_stream(self):
        """Test the streaming upload endpoint."""
        # Create a temporary PDF in memory
        pdf_data = b"%PDF-1.5\nTest PDF content"
        
        # Use mocks to avoid actual file operations and API calls
        with patch('routes.get_pdf_hash') as mock_hash, \
             patch('routes.preprocess_and_chunk') as mock_chunk, \
             patch('requests.post') as mock_post, \
             patch('builtins.open', create=True), \
             patch('json.dump'):
            
            # Setup mocks
            mock_hash.return_value = "test_hash"
            mock_chunk.return_value = [{"page_number": 0, "sentence_chunk": "Test sentence"}]
            
            # Mock API response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
            mock_post.return_value = mock_response
            
            # Test with valid PDF
            response = self.client.post(
                '/upload_stream',
                data={'file': (io.BytesIO(pdf_data), 'test.pdf')},
                content_type='multipart/form-data'
            )
            
            # Check if it's an event stream (using assertIn instead of assertEqual)
            self.assertIn('text/event-stream', response.content_type)
            
            # Test with no file
            response = self.client.post('/upload_stream')
            self.assertEqual(response.status_code, 400)
            
            # Test with non-PDF file
            response = self.client.post(
                '/upload_stream',
                data={'file': (io.BytesIO(b"Not a PDF"), 'test.txt')},
                content_type='multipart/form-data'
            )
            self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()