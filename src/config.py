"""
Configuration file for AI Study Assistant
Defines paths and model settings
"""
import os

class Config:
    # Base directory (project root)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Data storage paths
    UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")  # Path for uploaded documents
    FAISS_DB_DIR = os.path.join(BASE_DIR, "vector_db")      # Path for FAISS vector store
    
    # Model configurations
    SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
    QA_MODEL = "google/flan-t5-base"            # High-quality summarization (1.6GB)
    
    # Gemini API configuration
    GEMINI_API_KEY = "AIzaSyDPRDQFe0T85RQ1qcHnDi0uybTrDipqC-o" # Get from environment variable
    GEMINI_MODEL = "models/gemini-2.0-flash"                    # Updated to use 2.0 model
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"    
    # RAG Configuration
    RAG_CHUNK_SIZE = 1000           # Size of text chunks for processing
    RAG_CHUNK_OVERLAP = 200         # Overlap between chunks
    RAG_TOP_K = 5                   # Number of documents to retrieve
    RAG_TEMPERATURE = 0.2           # Temperature for generation
    RAG_MAX_OUTPUT_TOKENS = 2048    # Maximum tokens in response

    # Ollama configuration (set OLLAMA_ENABLED=True to use local LLM)
    OLLAMA_ENABLED = True
    OLLAMA_MODEL = "llama3.2:latest"
    OLLAMA_BASE_URL = "http://localhost:11434"
    # Optional: if you want to use Ollama for embeddings too
    OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
    
    # Ensure directories exist
    @staticmethod
    def create_directories():
        """Create necessary directories if they don't exist"""
        os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
        os.makedirs(Config.FAISS_DB_DIR, exist_ok=True)
