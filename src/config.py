"""Very simple config (kept small on purpose)."""
import os

class Config:
    # Basic folders (just make sure they exist)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    uploads_dir = os.path.join(root, "data", "uploads")
    db_dir = os.path.join(root, "vector_db")

    # Models (picked small ones so it runs faster)
    # Change these if you want different ones
    summarize_model = "facebook/bart-large-cnn"
    qa_model = "google/flan-t5-base"
    embedding_model = "sentence-transformers/all-mpnet-base-v2"

    # RAG settings (kept simple numbers)
    chunk_size = 800
    chunk_overlap = 150
    top_k = 5

    # If you want to use Gemini or Ollama, load keys from env instead of hardcoding
    gemini_key = os.environ.get("GEMINI_API_KEY", "")  # leave empty if not set
    gemini_model = "models/gemini-2.0-flash"

    # Local LLM via Ollama (optional). Set OLLAMA_ENABLED=1 in env to turn on.
    ollama_enabled = os.environ.get("OLLAMA_ENABLED", "0") == "1"
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    @staticmethod
    def make_dirs():
        for p in [Config.uploads_dir, Config.db_dir]:
            os.makedirs(p, exist_ok=True)

# Turn off heavy backends we don't use
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_JAX'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['USE_TF'] = '0'

# Create folders now
Config.make_dirs()
