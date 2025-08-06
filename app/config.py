import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # API Configuration
    API_TITLE = "100% Dynamic PDF Q&A System"
    API_VERSION = "13.0.0"
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # API Keys
    HACKRX_API_KEY = os.getenv("HACKRX_API_KEY", "hackrx-2024-api-key")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
    
    # Groq Models
    GROQ_MODELS = [
        "llama-3.1-8b-instant",       # Fastest
        "llama-3.1-70b-versatile",    # Most accurate
    ]
    
    # HuggingFace Model
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Production-optimized settings
    MAX_PDF_SIZE = 30 * 1024 * 1024  # Reduced for Render limits
    CHUNK_SIZE = 800                  # Reduced for memory efficiency
    CHUNK_OVERLAP = 150
    TOP_K_CHUNKS = 4                  # Reduced for speed
    REQUEST_TIMEOUT = 20
    MAX_RETRIES = 2

settings = Settings()
