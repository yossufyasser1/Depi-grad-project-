"""
Simple test script to demonstrate AI Study Assistant usage
Run this after starting main.py server
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print a section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_health():
    """Test health endpoint"""
    print_section("1. HEALTH CHECK")
    response = requests.get(f"{BASE_URL}/health")
    print(json.dumps(response.json(), indent=2))

def test_upload(pdf_path):
    """Test PDF upload"""
    print_section("2. UPLOAD PDF")
    try:
        with open(pdf_path, "rb") as f:
            files = {"file": ("document.pdf", f, "application/pdf")}
            response = requests.post(f"{BASE_URL}/upload", files=files)
            print(json.dumps(response.json(), indent=2))
    except FileNotFoundError:
        print(f"❌ File not found: {pdf_path}")
        print("Please update the pdf_path variable with a valid PDF file path")

def test_chat(question):
    """Test chat endpoint"""
    print_section("3. ASK QUESTION")
    data = {
        "question": question,
        "top_k": 5
    }
    response = requests.post(f"{BASE_URL}/chat", json=data)
    result = response.json()
    
    print(f"Question: {result['question']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nContext used: {result['context_used']}")
    print(f"Relevant chunks: {result['relevant_chunks']}")

def test_keywords(text):
    """Test keyword extraction"""
    print_section("4. EXTRACT KEYWORDS")
    data = {
        "text": text,
        "top_n": 10
    }
    response = requests.post(f"{BASE_URL}/keywords", json=data)
    result = response.json()
    
    print(f"Total words: {result['total_words']}\n")
    print("Top Keywords:")
    for i, kw in enumerate(result['keywords'], 1):
        print(f"  {i}. {kw['word']} - {kw['frequency']} times")

if __name__ == "__main__":
    print("\n🤖 AI Study Assistant - Test Script")
    print("Make sure the server is running at http://localhost:8000\n")
    
    # Test 1: Health check
    test_health()
    
    # Test 2: Upload PDF (UPDATE THIS PATH!)
    pdf_path = "your_document.pdf"  # ⚠️ CHANGE THIS TO YOUR PDF FILE
    # test_upload(pdf_path)  # Uncomment when you have a PDF
    
    # Test 3: Ask a question
    test_chat("What is the main topic of the document?")
    
    # Test 4: Extract keywords
    sample_text = """
    Artificial intelligence and machine learning are transforming 
    the way we process information. Natural language processing 
    enables computers to understand human language. Deep learning 
    models can analyze vast amounts of data efficiently.
    """
    test_keywords(sample_text)
    
    print("\n" + "="*60)
    print("✅ Tests Complete!")
    print("="*60 + "\n")
