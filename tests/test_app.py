import unittest
import sys
import os

# Füge das Projektverzeichnis zum Python-Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app  # Importiere die Flask-App

class TestApp(unittest.TestCase):
    def setUp(self):
        # Erstelle einen Testclient
        self.app = app.test_client()
        self.app.testing = True

    def test_index(self):
        # Teste die Hauptroute
        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)

    def test_chat_endpoint(self):
        # Teste den Chat-Endpoint
        response = self.app.post("/get", data={"msg": "Hello"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("response", response.json)