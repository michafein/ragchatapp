"""
Test cases for the Flask application routes and endpoints.
"""

import unittest
import sys
import os

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app  # Import the create_app factory function

class TestApp(unittest.TestCase):
    """Test class for Flask application endpoints."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create Flask app for testing
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for tests
        # Create a test client
        self.client = self.app.test_client()
        
    def test_index(self):
        """Test the main index route."""
        # Test the main route
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        # Check if the response contains HTML
        self.assertIn(b'<!DOCTYPE html>', response.data)

    def test_chat_endpoint(self):
        """Test the chat endpoint."""
        # Test the chat endpoint
        response = self.client.post("/get", data={"msg": "Hello"})
        self.assertEqual(response.status_code, 200)
        # Check if the response is a JSON with a response field
        json_data = response.get_json()
        self.assertIn("response", json_data)
        
    def test_clear_history_endpoint(self):
        """Test the clear history endpoint."""
        # Test the clear history endpoint
        response = self.client.post("/clear_history")
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn("message", json_data)
        self.assertEqual(json_data["message"], "Chat history cleared")

if __name__ == "__main__":
    unittest.main()