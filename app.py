import os
import logging
from flask import Flask
from config import Config

def create_app():
    """
    Application factory function to create and configure the Flask app.
    """
    app = Flask(__name__)
    app.config.from_object(Config)
    app.secret_key = Config.SECRET_KEY

    setup_logging()

    # Ensure required directories exist
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)
    os.makedirs("pages_and_chunks", exist_ok=True)

    # Import and register blueprints
    from routes import main_bp
    app.register_blueprint(main_bp)

    return app

def setup_logging():
    """
    Sets up logging configuration.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("chatbot.log"),
            logging.StreamHandler()
        ]
    )

if __name__ == '__main__':
    app = create_app()
    app.run(host="0.0.0.0", port=5000)
