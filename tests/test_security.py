"""
Test cases for security utilities.
"""

import unittest
from unittest.mock import MagicMock
import os
from utils.security import validate_file_upload, sanitize_input, set_secure_headers, validate_api_request

class TestSecurity(unittest.TestCase):
    """Test class for security utility functions."""

    def test_validate_file_upload(self):
        """Test file upload validation."""
        # Test valid PDF file
        mock_file = MagicMock()
        mock_file.filename = "document.pdf"
        mock_file.tell.return_value = 1000  # 1KB
        self.assertTrue(validate_file_upload(mock_file))
        
        # Test file with wrong extension
        mock_file.filename = "document.exe"
        self.assertFalse(validate_file_upload(mock_file))
        
        # Test file that's too large
        mock_file.filename = "document.pdf"
        mock_file.tell.return_value = 15 * 1024 * 1024  # 15MB
        self.assertFalse(validate_file_upload(mock_file))
        
        # Test empty filename
        mock_file.filename = ""
        self.assertFalse(validate_file_upload(mock_file))
        
        # Test None file
        self.assertFalse(validate_file_upload(None))

    def test_sanitize_input(self):
        """Test input sanitization."""
        # Test normal input
        self.assertEqual(sanitize_input("Hello World"), "Hello World")
        
        # Test script tag removal
        self.assertEqual(
            sanitize_input("<script>alert('XSS')</script>"), 
            ""
        )
        
        # Test complex script with attributes
        self.assertEqual(
            sanitize_input('<script type="text/javascript">alert("XSS")</script>'), 
            ""
        )
        
        # Test event handler removal
        input_with_events = '<a href="#" onclick="evil()">Click</a>'
        sanitized = sanitize_input(input_with_events)
        self.assertNotIn("onclick", sanitized)
        self.assertIn("<a href=", sanitized)
        
        # Test javascript: protocol
        sanitized = sanitize_input('<a href="javascript:alert(\'XSS\')">Click</a>')
        self.assertNotIn("javascript:", sanitized)
        self.assertIn("alert('XSS')", sanitized)
        
        # Test data URI - only check that data: is removed, not the exact format
        sanitized = sanitize_input('<img src="data:image/png;base64,AAAA">')
        self.assertNotIn("data:", sanitized)
        self.assertIn("<img src=", sanitized)
        self.assertIn("AAAA", sanitized)
        
        # Test non-string input
        self.assertEqual(sanitize_input(None), "")
        self.assertEqual(sanitize_input(123), "")

    def test_set_secure_headers(self):
        """Test secure headers setting."""
        # Create mock response
        mock_response = MagicMock()
        mock_response.headers = {}
        
        # Apply secure headers
        secured_response = set_secure_headers(mock_response)
        
        # Verify headers were set
        self.assertIn('Content-Security-Policy', secured_response.headers)
        self.assertIn('X-Content-Type-Options', secured_response.headers)
        self.assertIn('X-Frame-Options', secured_response.headers)
        self.assertIn('X-XSS-Protection', secured_response.headers)
        
        # Verify specific header values
        self.assertEqual(secured_response.headers['X-Content-Type-Options'], 'nosniff')
        self.assertEqual(secured_response.headers['X-Frame-Options'], 'DENY')
        self.assertEqual(secured_response.headers['X-XSS-Protection'], '1; mode=block')

    def test_validate_api_request(self):
        """Test API request validation."""
        # Valid request
        valid_request = {"message": "Hello", "timestamp": 123456}
        self.assertTrue(validate_api_request(valid_request))
        
        # Valid request with required fields
        self.assertTrue(validate_api_request(valid_request, ["message"]))
        
        # Invalid request (missing required field)
        self.assertFalse(validate_api_request(valid_request, ["missing_field"]))
        
        # Invalid request (not a dict)
        self.assertFalse(validate_api_request("not a dict"))
        
        # Invalid request (suspicious content)
        suspicious_request = {"message": "javascript:alert('XSS')"}
        self.assertFalse(validate_api_request(suspicious_request))
        
        # Another suspicious pattern
        suspicious_request = {"message": "<script>alert('XSS')</script>"}
        self.assertFalse(validate_api_request(suspicious_request))

if __name__ == "__main__":
    unittest.main()