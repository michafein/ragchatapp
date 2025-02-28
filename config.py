import os


class Config:
    # API URLs
    LM_STUDIO_API_URL = os.getenv("LM_STUDIO_API_URL", "http://host.docker.internal:1234/v1/chat/completions")
    LM_STUDIO_EMBEDDING_API_URL = os.getenv("LM_STUDIO_EMBEDDING_API_URL", "http://host.docker.internal:1234/v1/embeddings")
    
    # Model names
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-all-minilm-l6-v2-embedding")
    CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "deepseek-r1-distill-qwen-7b")
    
    # PDF path for default document (if needed)
    PDF_PATH = os.getenv("PDF_PATH", "human-nutrition-text.pdf")
    
    # Other settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    SECRET_KEY = os.getenv("SECRET_KEY", "RAG_my_key")  
    
    # RAG settings
    COSINE_SIMILARITY_THRESHOLD = float(os.getenv("COSINE_SIMILARITY_THRESHOLD", "0.6"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    
    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))