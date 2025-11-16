from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
from typing import Optional, List, Dict
from pathlib import Path
import logging
import tempfile
import os

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config import Config
from src.inference.chromadb_manager import ChromaDBManager
from src.inference.rag_retriever import RAGRetriever
from src.inference.llm_reader import LLMReader
from src.preprocessing.pdf_processor import PDFProcessor
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.training.textrank_summarizer import TextRankSummarizer
from src.training.keyword_extractor import KeywordExtractor

# Import schemas
from api.schemas import (
    ChatRequest, ChatResponse,
    SummarizeRequest, SummarizeResponse,
    KeywordRequest, KeywordResponse,
    UploadResponse, DocumentListResponse,
    HealthResponse, DeleteResponse,
    DocumentSource
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for loaded models
models = {}
config = Config()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load models on startup, cleanup on shutdown."""
    # Startup
    logger.info('=' * 50)
    logger.info('Starting AI Study Assistant API')
    logger.info('=' * 50)
    
    try:
        logger.info('Loading models and initializing components...')
        
        # Initialize vector store
        logger.info('1/5 Initializing ChromaDB...')
        models['vector_store'] = ChromaDBManager(
            collection_name='study_materials',
            config=config
        )
        
        # Initialize retriever
        logger.info('2/5 Initializing RAG Retriever...')
        models['retriever'] = RAGRetriever(
            vector_store_manager=models['vector_store'],
            config=config
        )
        
        # Initialize LLM (use smaller model for demo, can be changed)
        logger.info('3/5 Loading LLM (this may take a while)...')
        models['llm_reader'] = LLMReader(
            model_name='gpt2',  # Change to larger model if needed
            use_quantization=False,
            config=config
        )
        
        # Initialize processors
        logger.info('4/5 Initializing PDF and text processors...')
        models['pdf_processor'] = PDFProcessor()
        models['text_preprocessor'] = TextPreprocessor()
        
        # Initialize baseline models
        logger.info('5/5 Loading baseline models...')
        models['summarizer'] = TextRankSummarizer(num_sentences=3)
        models['keyword_extractor'] = KeywordExtractor()
        
        logger.info('✓ All models loaded successfully!')
        logger.info(f'Vector DB has {models["vector_store"].collection.count()} documents')
        logger.info('=' * 50)
        
    except Exception as e:
        logger.error(f'ERROR loading models: {e}')
        logger.error('Some endpoints may not work properly')
    
    yield
    
    # Shutdown
    logger.info('Shutting down AI Study Assistant API...')
    logger.info('Goodbye!')

# Initialize FastAPI app
app = FastAPI(
    title='AI Study Assistant API',
    description='Mini NotebookLM for document processing, RAG-based Q&A, and text analysis',
    version='1.0.0',
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

@app.get('/')
async def root():
    """Root endpoint with API information."""
    return {
        'name': 'AI Study Assistant API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'health': '/health',
            'upload': '/upload',
            'chat': '/chat',
            'summarize': '/summarize',
            'keywords': '/extract-keywords',
            'documents': '/documents'
        }
    }

@app.get('/health', response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    doc_count = 0
    if 'vector_store' in models:
        doc_count = models['vector_store'].collection.count()
    
    return HealthResponse(
        status='healthy',
        models_loaded=len(models),
        available_models=list(models.keys()),
        vector_db_documents=doc_count
    )

@app.post('/upload', response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document (PDF or text file)."""
    logger.info(f'Received upload request: {file.filename}')
    
    import time
    start_time = time.time()
    
    try:
        # Read file contents
        contents = await file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Process based on file type
            if file.filename.lower().endswith('.pdf'):
                logger.info('Processing PDF document...')
                
                # Process PDF
                result = models['pdf_processor'].process_pdf(
                    tmp_path,
                    str(config.PROCESSED_DATA_PATH / f"{file.filename}.json")
                )
                
                text = result['text']
                chunks = result['chunks']
                
            elif file.filename.lower().endswith(('.txt', '.md')):
                logger.info('Processing text document...')
                
                # Process text file
                text = contents.decode('utf-8')
                chunks = models['pdf_processor'].chunk_text(text)
                
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.filename}"
                )
            
            # Add to vector store
            logger.info(f'Adding {len(chunks)} chunks to vector database...')
            metadatas = [
                {
                    'source': file.filename,
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                }
                for i in range(len(chunks))
            ]
            ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]
            
            models['vector_store'].add_documents(chunks, metadatas, ids)
            
            logger.info(f'✓ Successfully processed {file.filename}')
            
            processing_time = time.time() - start_time
            
            return UploadResponse(
                upload_id=file.filename,
                filename=file.filename,
                status='success',
                chunks_processed=len(chunks),
                text_length=len(text),
                processing_time=processing_time,
                message=f'Successfully uploaded and indexed {file.filename}'
            )
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f'Upload error: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post('/chat', response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with documents using RAG (Retrieval-Augmented Generation)."""
    logger.info(f'Chat query: "{request.query}"')
    
    try:
        # Retrieve relevant documents
        if request.use_multi_query:
            retrieved_docs = models['retriever'].multi_query_retrieve(
                request.query,
                top_k=request.top_k
            )
        else:
            retrieved_docs = models['retriever'].retrieve_documents(
                request.query,
                top_k=request.top_k
            )
        
        if not retrieved_docs:
            return ChatResponse(
                query=request.query,
                answer='I could not find any relevant information in the uploaded documents.',
                sources=[],
                confidence=0.0,
                num_sources=0
            )
        
        # Generate answer using LLM
        result = models['llm_reader'].generate_with_retrieval(
            question=request.query,
            retrieved_docs=retrieved_docs,
            max_new_tokens=256,
            temperature=0.7
        )
        
        # Calculate average confidence
        avg_confidence = sum(doc['similarity'] for doc in retrieved_docs) / len(retrieved_docs)
        
        # Format sources
        sources = [
            DocumentSource(
                text=doc['text'][:200] + '...' if len(doc['text']) > 200 else doc['text'],
                metadata=doc['metadata'],
                similarity=doc['similarity'],
                id=doc.get('id')
            )
            for doc in retrieved_docs
        ]
        
        return ChatResponse(
            query=request.query,
            answer=result['answer'],
            sources=sources,
            confidence=avg_confidence,
            num_sources=len(retrieved_docs)
        )
        
    except Exception as e:
        logger.error(f'Chat error: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post('/summarize', response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    """Summarize text using extractive or abstractive methods."""
    logger.info(f'Summarizing {len(request.text)} characters ({request.summary_type})')
    
    try:
        if request.summary_type == 'extractive':
            # Use TextRank for extractive summarization
            num_sentences = request.num_sentences if request.num_sentences is not None else 3
            summarizer = TextRankSummarizer(num_sentences=num_sentences)
            summary = summarizer.summarize(request.text)
            method = 'TextRank (Extractive)'
            
        else:
            # Use LLM for abstractive summarization
            prompt = f"Summarize the following text concisely:\n\n{request.text}\n\nSummary:"
            summary = models['llm_reader'].generate_answer(
                question="Summarize this text",
                context=request.text,
                max_new_tokens=150,
                temperature=0.5,
                template='concise'
            )
            method = 'LLM (Abstractive)'
        
        compression_ratio = len(summary) / len(request.text) if len(request.text) > 0 else 0
        
        return SummarizeResponse(
            original_length=len(request.text),
            summary=summary,
            summary_length=len(summary),
            summary_type=request.summary_type,
            method=method,
            compression_ratio=round(compression_ratio, 3)
        )
        
    except Exception as e:
        logger.error(f'Summarization error: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.post('/extract-keywords', response_model=KeywordResponse)
async def extract_keywords(request: KeywordRequest):
    """Extract keywords from text using RAKE algorithm."""
    logger.info(f'Extracting {request.top_n} keywords from {len(request.text)} characters')
    
    try:
        keywords = models['keyword_extractor'].extract_keywords(
            request.text,
            top_n=request.top_n
        )
        
        from api.schemas import KeywordItem
        
        return KeywordResponse(
            text_length=len(request.text),
            text_preview=request.text[:100] + '...' if len(request.text) > 100 else request.text,
            keywords=[
                KeywordItem(keyword=keyword, score=round(score, 4))
                for keyword, score in keywords
            ],
            num_keywords=len(keywords)
        )
        
    except Exception as e:
        logger.error(f'Keyword extraction error: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f"Keyword extraction failed: {str(e)}")

@app.get('/documents', response_model=DocumentListResponse)
async def list_documents():
    """List uploaded documents and collection statistics."""
    try:
        stats = models['vector_store'].get_collection_stats()
        
        # Get sample documents
        sample = models['vector_store'].peek(limit=5)
        
        sample_docs = [
            {
                'id': sample['ids'][i] if i < len(sample['ids']) else None,
                'preview': sample['documents'][i][:100] + '...' if i < len(sample['documents']) else None,
                'metadata': sample['metadatas'][i] if i < len(sample['metadatas']) else None
            }
            for i in range(min(3, len(sample.get('ids', []))))
        ]
        
        return DocumentListResponse(
            total_documents=stats['document_count'],
            collection_name=stats['collection_name'],
            embedding_dimension=stats['embedding_dimension'],
            embedding_model=stats['embedding_model'],
            sample_documents=sample_docs
        )
        
    except Exception as e:
        logger.error(f'List documents error: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.delete('/documents/{document_id}', response_model=DeleteResponse)
async def delete_document(document_id: str):
    """Delete a specific document by ID."""
    logger.info(f'Deleting document: {document_id}')
    
    try:
        # Find all chunks for this document
        all_data = models['vector_store'].collection.get()
        
        ids_to_delete = [
            id for id in all_data['ids']
            if id.startswith(document_id)
        ]
        
        if not ids_to_delete:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        models['vector_store'].delete_documents(ids_to_delete)
        
        return DeleteResponse(
            status='success',
            document_id=document_id,
            chunks_deleted=len(ids_to_delete),
            message=f'Successfully deleted {document_id}'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Delete error: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(
        'main:app',
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level=config.LOG_LEVEL.lower()
    )
