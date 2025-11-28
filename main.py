"""
AI Study Assistant - Main FastAPI Application
Provides endpoints for PDF upload, chat, and keyword extraction
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import shutil
from typing import Optional

# Import our custom classes
from src.config import Config
from src.processing.pdf_processor import PDFProcessor
from src.processing.improved_rag_retriever import ImprovedRAGRetriever
from src.inference.improved_llm_reader import ImprovedLLMReader
from src.inference.qa_generator import QAGenerator
from src.inference.summarizer import Summarizer


# Pydantic models for request/response validation
class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    question: str
    top_k: Optional[int] = 5


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    question: str
    answer: str
    context_used: bool
    relevant_chunks: int


class GenerateQARequest(BaseModel):
    """Request model for Q&A generation from text"""
    text: str
    num_questions: Optional[int] = 10


class QAResponse(BaseModel):
    """Response model for Q&A generation"""
    total_qa_pairs: int
    qa_pairs: list
    source_length: int
    status: str


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    message: str
    document_count: int


class SummarizeRequest(BaseModel):
    """Request model for text summarization"""
    text: str
    max_length: Optional[int] = 200
    min_length: Optional[int] = 50


class SummarizeResponse(BaseModel):
    """Response model for summarization"""
    summary: str
    original_length: int
    summary_length: int


# Initialize FastAPI app
app = FastAPI(title="AI Study Assistant API", version="1.0.0")

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Mount static files (frontend)
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Global variables for our components (initialized on startup)
pdf_processor = None
rag_retriever = None
llm_reader = None
qa_generator = None
summarizer = None


@app.on_event("startup")
async def startup_event():
    """Initialize core components on startup."""
    global pdf_processor, rag_retriever, llm_reader, qa_generator, summarizer
    
    print("\n🚀 Starting AI Study Assistant")
    
    # Create necessary directories
    Config.create_directories()
    
    # Initialize all components
    print("📦 Initializing components...")
    
    pdf_processor = PDFProcessor(chunk_size=500)
    rag_retriever = ImprovedRAGRetriever()  # Enhanced RAG with FAISS
    llm_reader = ImprovedLLMReader()  # Enhanced LLM with Gemini 2.0
    qa_generator = QAGenerator()
    summarizer = Summarizer()  # Local summarization model
    
    print("✅ Backend ready (FAISS + Gemini)\n")


@app.get("/", tags=["General"])
async def root():
    """Serve the web interface or a simple index."""
    frontend_file = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if os.path.exists(frontend_file):
        return FileResponse(frontend_file)
    else:
        return {
            "message": "Welcome to AI Study Assistant API",
            "endpoints": {
                "upload": "POST /upload - Upload PDF files",
                "upload_with_qa": "POST /upload-with-qa - Upload PDF and generate Q&A",
                "chat": "POST /chat - Ask questions",
                "keywords": "POST /keywords - Extract keywords",
                "generate_qa": "POST /generate-qa - Generate Q&A from text",
                "health": "GET /health - Check system health"
            },
            "ui": "Frontend files not found. Please check the frontend directory."
        }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Check system health and get document count
    """
    try:
        stats = rag_retriever.get_database_stats()
        doc_count = stats.get('document_count', 0)
        print(f"✅ System healthy - Enhanced RAG: {doc_count} documents")
        
        return HealthResponse(
            status="healthy",
            message=f"Enhanced RAG system operational - {doc_count} documents indexed",
            document_count=doc_count
        )
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"System error: {str(e)}")


@app.post("/upload", tags=["Documents"])
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, extract text, chunk it, and index in FAISS."""
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        print("❌ Invalid file type\n")
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file temporarily
        upload_path = os.path.join(Config.UPLOAD_DIR, file.filename)
        
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        text, chunks = pdf_processor.process_pdf(upload_path)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Add to RAG (FAISS)
        success = rag_retriever.load_documents()
        
        if success:
            chunks_added = len(chunks)
        else:
            raise HTTPException(status_code=500, detail="Failed to add documents to RAG system")
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_added": chunks_added,
            "total_documents_in_db": rag_retriever.get_database_stats().get('document_count', 0),
            "message": f"Successfully processed {file.filename} with Enhanced RAG"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """Answer a question using FAISS + Gemini, or Gemini alone."""
    
    try:
        stats = rag_retriever.get_database_stats()
        if stats['document_count'] > 0:
            context = rag_retriever.retrieve_and_format(request.question, top_k=request.top_k)
            
            if context:
                answer = llm_reader.generate_answer(request.question, context)
                relevant_docs = rag_retriever.retrieve(request.question, top_k=request.top_k)
                
                print(f"✅ Enhanced RAG answer generated successfully\n")
                
                return ChatResponse(
                    question=request.question,
                    answer=answer,
                    context_used=True,
                    relevant_chunks=len(relevant_docs)
                )
        
        # No documents available
        print("⚠️ No documents in database - generating answer without context\n")
        
        # Generate answer without context
        answer = llm_reader.generate_simple_answer(request.question)
        
        return ChatResponse(
            question=request.question,
            answer=answer,
            context_used=False,
            relevant_chunks=0
        )
        
    except Exception as e:
        print(f"❌ Chat failed: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/summarize", response_model=SummarizeResponse, tags=["Summarization"])
async def summarize_text(request: SummarizeRequest):
    """Summarize text using BART-based summarizer."""
    
    try:
        # Generate summary using dedicated Summarizer (DistilBART)
        summary = summarizer.summarize_text(
            request.text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        
        print(f"✅ Summary generated: {len(summary)} characters\n")
        
        return SummarizeResponse(
            summary=summary,
            original_length=len(request.text),
            summary_length=len(summary)
        )
        
    except Exception as e:
        print(f"❌ Summarization failed: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@app.post("/generate-qa", response_model=QAResponse, tags=["Q&A Generation"])
async def generate_qa(request: GenerateQARequest):
    """Generate Q&A pairs from provided text."""
    
    try:
        # Generate Q&A pairs
        result = qa_generator.generate_qa_from_pdf_text(
            request.text,
            num_questions=request.num_questions
        )
        
        print(f"✅ Generated {result['total_qa_pairs']} Q&A pairs\n")
        
        return QAResponse(
            total_qa_pairs=result['total_qa_pairs'],
            qa_pairs=result['qa_pairs'],
            source_length=result['source_length'],
            status=result['status']
        )
        
    except Exception as e:
        print(f"❌ Q&A generation failed: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Q&A generation failed: {str(e)}")


@app.post("/upload-with-qa", tags=["Documents"])
async def upload_pdf_with_qa(file: UploadFile = File(...), num_questions: int = 10):
    """Upload a PDF, extract text, generate Q&A, and index in FAISS."""
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        print("❌ Invalid file type\n")
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file
        upload_path = os.path.join(Config.UPLOAD_DIR, file.filename)
        
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process PDF
        text, chunks = pdf_processor.process_pdf(upload_path)
        
        if not chunks:
            print("❌ No text extracted from PDF\n")
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Generate Q&A pairs from the full text
        qa_result = qa_generator.generate_qa_from_pdf_text(text, num_questions=num_questions)
        
        # Add to RAG (FAISS)
        success = rag_retriever.load_documents()
        
        if success:
            print("✅ Documents added to Enhanced RAG successfully")
            chunks_added = len(chunks)
        else:
            print("❌ Enhanced RAG update failed")
            raise HTTPException(status_code=500, detail="Failed to add documents to RAG system")
        
        print(f"✅ Upload successful: {file.filename}")
        print(f"   Added {chunks_added} chunks to database")
        print(f"   Generated {qa_result['total_qa_pairs']} Q&A pairs\n")
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_added": chunks_added,
            "total_documents_in_db": rag_retriever.get_database_stats().get('document_count', 0),
            "qa_pairs": qa_result['qa_pairs'],
            "total_qa_generated": qa_result['total_qa_pairs'],
            "full_text": text,  # Include full text for frontend summarization
            "message": f"Successfully processed {file.filename} and generated {qa_result['total_qa_pairs']} Q&A pairs with Enhanced RAG"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload with Q&A failed: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# Additional utility endpoints
@app.get("/rag-stats", tags=["General"])
async def get_rag_stats():
    """Get statistics about the RAG system and conversation."""
    try:
        # Enhanced RAG stats
        enhanced_stats = rag_retriever.get_database_stats()
        
        # LLM stats
        llm_stats = llm_reader.get_conversation_stats()
        
        return {
            "enhanced_rag": enhanced_stats,
            "conversation": llm_stats,
            "system_status": "operational",
            "database_type": "FAISS + Google Embeddings"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


@app.delete("/documents", tags=["Documents"])
async def clear_documents():
    """Clear all documents from the FAISS database (use with caution)."""
    
    try:
        # Clear enhanced RAG
        success = rag_retriever.reset_database()
        
        if success:
            print("✅ Enhanced RAG database cleared")
        else:
            print("⚠️ Enhanced RAG clear failed")
        
        print("✅ All documents deleted\n")
        
        return {
            "status": "success",
            "message": "All documents cleared from Enhanced RAG system",
            "document_count": 0
        }
        
    except Exception as e:
        print(f"❌ Delete failed: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("\n🚀 Starting AI Study Assistant API...")
    print("📍 http://localhost:8000 | Docs: /docs\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
