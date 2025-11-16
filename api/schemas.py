from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime

# ==================== REQUEST MODELS ====================

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str = Field(..., description='User query about the documents', min_length=1)
    top_k: int = Field(5, description='Number of documents to retrieve', ge=1, le=20)
    similarity_threshold: float = Field(0.5, description='Minimum similarity score', ge=0.0, le=1.0)
    use_multi_query: bool = Field(False, description='Use multi-query retrieval for better results')
    
    @validator('query')
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "top_k": 5,
                "similarity_threshold": 0.5,
                "use_multi_query": False
            }
        }


class SummarizeRequest(BaseModel):
    """Request model for summarization endpoint."""
    text: str = Field(..., description='Text to summarize', min_length=50)
    summary_type: str = Field('abstractive', description='extractive or abstractive')
    max_length: Optional[int] = Field(256, description='Maximum summary length', ge=50, le=1024)
    num_sentences: Optional[int] = Field(3, description='Number of sentences (for extractive)', ge=1, le=10)
    
    @validator('summary_type')
    def validate_summary_type(cls, v):
        allowed = ['extractive', 'abstractive']
        if v.lower() not in allowed:
            raise ValueError(f'summary_type must be one of {allowed}')
        return v.lower()
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Long text to be summarized...",
                "summary_type": "abstractive",
                "max_length": 256,
                "num_sentences": 3
            }
        }


class KeywordRequest(BaseModel):
    """Request model for keyword extraction endpoint."""
    text: str = Field(..., description='Text to extract keywords from', min_length=20)
    top_n: int = Field(10, description='Number of keywords to extract', ge=1, le=50)
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Artificial intelligence and machine learning are transforming technology...",
                "top_n": 10
            }
        }


class QuizRequest(BaseModel):
    """Request model for quiz generation endpoint."""
    text: str = Field(..., description='Text to generate quiz from', min_length=100)
    num_questions: int = Field(5, description='Number of questions', ge=1, le=20)
    difficulty_level: str = Field('medium', description='easy, medium, or hard')
    question_type: Optional[str] = Field('multiple_choice', description='Type of questions to generate')
    
    @validator('difficulty_level')
    def validate_difficulty(cls, v):
        allowed = ['easy', 'medium', 'hard']
        if v.lower() not in allowed:
            raise ValueError(f'difficulty_level must be one of {allowed}')
        return v.lower()
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Study material text for quiz generation...",
                "num_questions": 5,
                "difficulty_level": "medium",
                "question_type": "multiple_choice"
            }
        }


class UploadRequest(BaseModel):
    """Request model for file upload metadata."""
    filename: str = Field(..., description='Name of uploaded file')
    file_type: str = Field('pdf', description='Type of file: pdf, txt, or md')
    
    @validator('file_type')
    def validate_file_type(cls, v):
        allowed = ['pdf', 'txt', 'md']
        if v.lower() not in allowed:
            raise ValueError(f'file_type must be one of {allowed}')
        return v.lower()


# ==================== RESPONSE MODELS ====================

class DocumentSource(BaseModel):
    """Document source information for retrieved documents."""
    text: str = Field(..., description='Document text or excerpt')
    metadata: Dict[str, Any] = Field(default_factory=dict, description='Document metadata')
    similarity: float = Field(..., description='Similarity score', ge=0.0, le=1.0)
    id: Optional[str] = Field(None, description='Document ID in vector store')
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Machine learning is a subset of artificial intelligence...",
                "metadata": {"source": "ai_textbook.pdf", "chunk_id": 5},
                "similarity": 0.87,
                "id": "ai_textbook.pdf_chunk_5"
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    query: str = Field(..., description='Original user query')
    answer: str = Field(..., description='Generated answer')
    sources: List[DocumentSource] = Field(default_factory=list, description='Retrieved source documents')
    confidence: float = Field(..., description='Answer confidence score', ge=0.0, le=1.0)
    num_sources: int = Field(..., description='Number of sources used')
    timestamp: datetime = Field(default_factory=datetime.now, description='Response timestamp')
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "answer": "Machine learning is a subset of AI that enables systems to learn...",
                "sources": [],
                "confidence": 0.85,
                "num_sources": 3,
                "timestamp": "2025-11-17T10:30:00"
            }
        }


class SummarizeResponse(BaseModel):
    """Response model for summarization endpoint."""
    original_length: int = Field(..., description='Length of original text')
    summary: str = Field(..., description='Generated summary')
    summary_length: int = Field(..., description='Length of summary')
    summary_type: str = Field(..., description='Type of summarization used')
    method: str = Field(..., description='Summarization method/model used')
    compression_ratio: float = Field(..., description='Compression ratio (summary_length/original_length)')
    
    class Config:
        json_schema_extra = {
            "example": {
                "original_length": 1500,
                "summary": "This is a concise summary of the original text...",
                "summary_length": 150,
                "summary_type": "abstractive",
                "method": "LLM (Abstractive)",
                "compression_ratio": 0.1
            }
        }


class KeywordItem(BaseModel):
    """Individual keyword with score."""
    keyword: str = Field(..., description='Extracted keyword or phrase')
    score: float = Field(..., description='Keyword importance score', ge=0.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "keyword": "machine learning",
                "score": 8.5
            }
        }


class KeywordResponse(BaseModel):
    """Response model for keyword extraction endpoint."""
    text_length: int = Field(..., description='Length of input text')
    text_preview: str = Field(..., description='Preview of input text')
    keywords: List[KeywordItem] = Field(..., description='Extracted keywords with scores')
    num_keywords: int = Field(..., description='Number of keywords extracted')
    
    class Config:
        json_schema_extra = {
            "example": {
                "text_length": 500,
                "text_preview": "Artificial intelligence and machine learning...",
                "keywords": [
                    {"keyword": "machine learning", "score": 8.5},
                    {"keyword": "artificial intelligence", "score": 7.2}
                ],
                "num_keywords": 10
            }
        }


class QuizQuestion(BaseModel):
    """Individual quiz question."""
    question: str = Field(..., description='Question text')
    options: Optional[List[str]] = Field(None, description='Multiple choice options')
    correct_answer: str = Field(..., description='Correct answer')
    explanation: Optional[str] = Field(None, description='Explanation of the answer')
    difficulty: str = Field(..., description='Question difficulty level')


class QuizResponse(BaseModel):
    """Response model for quiz generation endpoint."""
    text_preview: str = Field(..., description='Preview of source text')
    questions: List[QuizQuestion] = Field(..., description='Generated quiz questions')
    num_questions: int = Field(..., description='Number of questions generated')
    difficulty_level: str = Field(..., description='Quiz difficulty level')


class UploadResponse(BaseModel):
    """Response model for file upload endpoint."""
    upload_id: str = Field(..., description='Unique upload identifier')
    filename: str = Field(..., description='Name of uploaded file')
    status: str = Field(..., description='Upload status')
    chunks_processed: int = Field(..., description='Number of chunks created')
    text_length: int = Field(..., description='Total length of extracted text')
    processing_time: Optional[float] = Field(None, description='Processing time in seconds')
    message: str = Field(..., description='Status message')
    
    class Config:
        json_schema_extra = {
            "example": {
                "upload_id": "document123.pdf",
                "filename": "document123.pdf",
                "status": "success",
                "chunks_processed": 42,
                "text_length": 15000,
                "processing_time": 2.5,
                "message": "Successfully uploaded and indexed document123.pdf"
            }
        }


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    total_documents: int = Field(..., description='Total number of document chunks')
    collection_name: str = Field(..., description='Vector database collection name')
    embedding_dimension: int = Field(..., description='Embedding vector dimension')
    embedding_model: str = Field(..., description='Embedding model name')
    sample_documents: List[Dict[str, Any]] = Field(default_factory=list, description='Sample documents')
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_documents": 150,
                "collection_name": "study_materials",
                "embedding_dimension": 384,
                "embedding_model": "all-MiniLM-L6-v2",
                "sample_documents": []
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description='API health status')
    models_loaded: int = Field(..., description='Number of models loaded')
    available_models: List[str] = Field(default_factory=list, description='List of loaded models')
    vector_db_documents: int = Field(..., description='Number of documents in vector DB')
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "models_loaded": 6,
                "available_models": ["vector_store", "retriever", "llm_reader"],
                "vector_db_documents": 150
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description='Error type')
    message: str = Field(..., description='Error message')
    detail: Optional[str] = Field(None, description='Detailed error information')
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input",
                "detail": "Query cannot be empty"
            }
        }


class DeleteResponse(BaseModel):
    """Response model for document deletion."""
    status: str = Field(..., description='Deletion status')
    document_id: str = Field(..., description='ID of deleted document')
    chunks_deleted: int = Field(..., description='Number of chunks deleted')
    message: str = Field(..., description='Status message')
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "document_id": "document123.pdf",
                "chunks_deleted": 42,
                "message": "Successfully deleted document123.pdf"
            }
        }
