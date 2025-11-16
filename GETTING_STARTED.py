"""
=================================================================
AI STUDY ASSISTANT - QUICK START GUIDE
=================================================================

This guide will help you get started with the AI Study Assistant.
Follow the steps below to set up, test, and use the system.

=================================================================
"""

# =================================================================
# STEP 1: INSTALLATION & SETUP
# =================================================================

"""
1. Install dependencies:
   pip install -r requirements.txt

2. Download required models:
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

3. Download datasets (optional, for training):
   python src/preprocessing/download_datasets.py

4. Configure environment:
   cp .env.example .env
   # Edit .env with your settings
"""

# =================================================================
# STEP 2: START THE API
# =================================================================

"""
Option A - Local Development:
   python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Option B - Docker (Recommended):
   docker-compose up -d
   
   # Check logs
   docker logs -f ai-study-assistant
   
   # Stop
   docker-compose down
"""

# =================================================================
# STEP 3: VERIFY INSTALLATION
# =================================================================

"""
1. Check API health:
   curl http://localhost:8000/health
   
   Expected response:
   {"status": "ok", "timestamp": "..."}

2. View API documentation:
   Open browser: http://localhost:8000/docs
   
3. Test basic endpoint:
   curl -X POST "http://localhost:8000/extract-keywords?text=Machine%20learning%20is%20amazing&top_n=5"
"""

# =================================================================
# STEP 4: USING THE API
# =================================================================

print("\n" + "="*60)
print("API ENDPOINTS GUIDE")
print("="*60)

endpoints = {
    "Health Check": {
        "method": "GET",
        "endpoint": "/health",
        "description": "Check API status",
        "example": 'curl http://localhost:8000/health'
    },
    "Upload Document": {
        "method": "POST",
        "endpoint": "/upload",
        "description": "Upload PDF document",
        "example": 'curl -X POST -F "file=@document.pdf" http://localhost:8000/upload'
    },
    "Chat/Q&A": {
        "method": "POST",
        "endpoint": "/chat",
        "description": "Ask questions about documents",
        "example": 'curl -X POST "http://localhost:8000/chat?query=What%20is%20AI&top_k=5"'
    },
    "Summarize Text": {
        "method": "POST",
        "endpoint": "/summarize",
        "description": "Generate text summary",
        "example": 'curl -X POST "http://localhost:8000/summarize?text=Your%20text%20here&summary_type=extractive"'
    },
    "Extract Keywords": {
        "method": "POST",
        "endpoint": "/extract-keywords",
        "description": "Extract keywords from text",
        "example": 'curl -X POST "http://localhost:8000/extract-keywords?text=Your%20text&top_n=10"'
    },
    "List Documents": {
        "method": "GET",
        "endpoint": "/documents",
        "description": "Get all uploaded documents",
        "example": 'curl http://localhost:8000/documents'
    },
    "Delete Document": {
        "method": "DELETE",
        "endpoint": "/documents/{doc_id}",
        "description": "Delete specific document",
        "example": 'curl -X DELETE http://localhost:8000/documents/doc123'
    }
}

for name, details in endpoints.items():
    print(f"\n{name}:")
    print(f"  Method: {details['method']}")
    print(f"  Endpoint: {details['endpoint']}")
    print(f"  Description: {details['description']}")
    print(f"  Example: {details['example']}")

# =================================================================
# STEP 5: PYTHON CLIENT USAGE
# =================================================================

print("\n" + "="*60)
print("PYTHON CLIENT EXAMPLES")
print("="*60)

python_examples = """
# Import requests library
import requests

BASE_URL = "http://localhost:8000"

# 1. Health Check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# 2. Upload Document
with open("document.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(f"{BASE_URL}/upload", files=files)
    print(response.json())

# 3. Ask Question
params = {
    "query": "What is machine learning?",
    "top_k": 5
}
response = requests.post(f"{BASE_URL}/chat", params=params)
result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])} documents")

# 4. Summarize Text
params = {
    "text": "Your long text here...",
    "summary_type": "extractive"  # or "abstractive"
}
response = requests.post(f"{BASE_URL}/summarize", params=params)
print(f"Summary: {response.json()['summary']}")

# 5. Extract Keywords
params = {
    "text": "Machine learning and AI are transforming industries.",
    "top_n": 5
}
response = requests.post(f"{BASE_URL}/extract-keywords", params=params)
keywords = response.json()["keywords"]
for keyword, score in keywords:
    print(f"  {keyword}: {score:.3f}")

# 6. List Documents
response = requests.get(f"{BASE_URL}/documents")
data = response.json()
print(f"Total documents: {data['total_documents']}")
for doc in data['documents'][:5]:  # Show first 5
    print(f"  - {doc['id']}: {doc.get('source', 'unknown')}")
"""

print(python_examples)

# =================================================================
# STEP 6: MODELS & CONFIGURATION
# =================================================================

print("\n" + "="*60)
print("MODELS USED")
print("="*60)

models = {
    "Summarization": "facebook/bart-large-cnn (BART)",
    "NER": "distilbert-base-uncased (DistilBERT)",
    "Question Answering": "distilbert-base-uncased-distilled-squad",
    "Embeddings": "all-MiniLM-L6-v2 (sentence-transformers)",
    "LLM": "gpt2 (GPT-2)",
    "Vector Database": "ChromaDB",
    "Fine-tuning": "T5-base with LoRA"
}

for task, model in models.items():
    print(f"{task:.<30} {model}")

# =================================================================
# STEP 7: PERFORMANCE EXPECTATIONS
# =================================================================

print("\n" + "="*60)
print("PERFORMANCE METRICS")
print("="*60)

performance = {
    "Average Latency": "< 2 seconds",
    "Document Retrieval": "< 500ms",
    "Summarization (Extractive)": "< 1 second",
    "Summarization (Abstractive)": "< 5 seconds",
    "Keyword Extraction": "< 500ms",
    "Throughput": "50 requests/min per worker",
    "Memory Usage": "~4GB with all models loaded",
    "Concurrent Requests": "> 10 simultaneous"
}

for metric, value in performance.items():
    print(f"{metric:.<30} {value}")

# =================================================================
# STEP 8: TROUBLESHOOTING
# =================================================================

print("\n" + "="*60)
print("COMMON ISSUES & SOLUTIONS")
print("="*60)

issues = [
    {
        "issue": "Port 8000 already in use",
        "solution": "Change API_PORT in .env or stop other service using: netstat -ano | findstr :8000"
    },
    {
        "issue": "Models not found",
        "solution": "Run: python src/preprocessing/download_datasets.py"
    },
    {
        "issue": "Tesseract not found",
        "solution": "Install tesseract-ocr: apt-get install tesseract-ocr (Linux) or brew install tesseract (Mac)"
    },
    {
        "issue": "Out of memory",
        "solution": "Reduce BATCH_SIZE in .env or enable quantization"
    },
    {
        "issue": "CUDA errors",
        "solution": "Set USE_CUDA=False in .env to use CPU"
    }
]

for idx, item in enumerate(issues, 1):
    print(f"\n{idx}. {item['issue']}")
    print(f"   Solution: {item['solution']}")

# =================================================================
# STEP 9: TESTING
# =================================================================

print("\n" + "="*60)
print("RUNNING TESTS")
print("="*60)

test_commands = """
# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/test_unit.py -v

# Run integration tests only
pytest tests/test_integration.py -v

# Run with coverage
pytest --cov=src --cov-report=html tests/

# View coverage report
# Open: htmlcov/index.html
"""

print(test_commands)

# =================================================================
# STEP 10: PERFORMANCE PROFILING
# =================================================================

print("\n" + "="*60)
print("PERFORMANCE PROFILING")
print("="*60)

profiling_commands = """
# Run full benchmark suite
python src/performance_profiler.py

# Or use Python
from src.performance_profiler import run_full_benchmark
results = run_full_benchmark()

# Check system resources
from src.performance_profiler import system_resources
system_resources()

# Benchmark specific components
from src.performance_profiler import benchmark_summarization
benchmark_summarization()
"""

print(profiling_commands)

# =================================================================
# NEXT STEPS
# =================================================================

print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)

next_steps = """
1. ✅ API is running and tested
2. 📚 Upload your first document via /upload
3. 💬 Try asking questions via /chat
4. 📝 Test summarization with your content
5. 🔍 Experiment with keyword extraction
6. 🧪 Run the test suite to verify everything works
7. 📊 Use MLflow to track experiments
8. 🚀 Deploy to production using Docker
9. 📈 Monitor performance with profiler
10. 🎨 Consider building a frontend UI

For more details, see:
- README.md - Complete documentation
- DEPLOYMENT_CHECKLIST.md - Deployment guide
- http://localhost:8000/docs - Interactive API docs
"""

print(next_steps)

print("\n" + "="*60)
print("🎉 YOU'RE READY TO GO!")
print("="*60)
print("\nVisit http://localhost:8000/docs to explore the interactive API documentation!")
