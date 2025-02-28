"""
RAG Chatbot Application.
Main application file with backward compatibility.
"""

import os
import logging
from flask import Flask

# Import configuration
from config import Config

# For backward compatibility, create a global app variable
app = None

def create_app():
    """
    Application factory function to create and configure the Flask app.
    
    Returns:
        Flask: Configured Flask application instance
    """
    global app
    app = Flask(__name__)
    app.config.from_object(Config)
    app.secret_key = Config.SECRET_KEY

    # Set up logging
    setup_logging()

    # Ensure required directories exist
    for directory in ["uploads", "embeddings", "pages_and_chunks"]:
        os.makedirs(directory, exist_ok=True)

    # Import and register blueprints
    from routes import main_bp
    app.register_blueprint(main_bp)

    app.logger.info("Application initialized successfully")
    return app

def setup_logging():
    """
    Sets up logging configuration.
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if Config.DEBUG else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/chatbot.log"),
            logging.StreamHandler()
        ]
    )
    
    # Reduce verbosity of some libraries
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

# Create app instance for backward compatibility
app = create_app()

if __name__ == '__main__':
    # No need to create app again as it's already created above
    app.run(host="0.0.0.0", port=5000)