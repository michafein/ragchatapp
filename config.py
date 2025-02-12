import os

#from dotenv import load_dotenv
#load_dotenv()

class Config:
    # API URLs
    LM_STUDIO_API_URL = os.getenv("LM_STUDIO_API_URL", "http://host.docker.internal:1234/v1/chat/completions")
    LM_STUDIO_EMBEDDING_API_URL = os.getenv("LM_STUDIO_EMBEDDING_API_URL", "http://host.docker.internal:1234/v1/embeddings")
    
    # Model names
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-all-minilm-l6-v2-embedding")
    CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "deepseek-r1-distill-qwen-7b")
    
    # PDF path
    PDF_PATH = os.getenv("PDF_PATH", "human-nutrition-text.pdf")
    
    # Other settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

