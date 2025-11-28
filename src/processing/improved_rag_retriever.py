"""
Improved RAG Retriever - Enhanced Retrieval-Augmented Generation component
Uses FAISS vector store with Google Gemini embeddings for better performance
"""
import os
import logging
import glob
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

# LangChain imports
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)

# Local imports
from src.config import Config

logger = logging.getLogger(__name__)

class ImprovedRAGRetriever:
    """Enhanced RAG retriever using FAISS and Google embeddings"""
    
    def __init__(self):
        """
        Initialize the improved RAG retriever.
        """
        print("🚀 Initializing Improved RAG Retriever...")
        
        # Configuration
        self.docs_dir = Config.UPLOAD_DIR
        self.cache_dir = Config.FAISS_DB_DIR
        self.chunk_size = Config.RAG_CHUNK_SIZE
        self.chunk_overlap = Config.RAG_CHUNK_OVERLAP
        self.top_k = Config.RAG_TOP_K
        
        # Create necessary directories
        Config.create_directories()
        
        # Initialize components
        self.vector_db = None
        self.embeddings = None
        self.session_id = str(uuid.uuid4())
        
        # Initialize embeddings
        self._init_embeddings()
        
        # Try to load existing database
        if not self.load_existing_db():
            print("📁 No existing vector database found. Will create when documents are added.")
        
        print("✅ Improved RAG Retriever ready!\n")
    
    def _init_embeddings(self) -> None:
        """Initialize local sentence-transformers embeddings (no API quota)."""
        print("🔧 Initializing local sentence-transformers embeddings...")
        try:
            model_name = Config.EMBEDDING_MODEL
            self.embeddings = SentenceTransformerEmbeddings(model_name=model_name)
            print(f"✅ Embeddings initialized: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def _get_document_loader(self, file_path: str):
        """Get appropriate document loader based on file type."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return PyPDFLoader(file_path)
        elif ext == '.docx':
            return Docx2txtLoader(file_path)
        elif ext == '.txt':
            return TextLoader(file_path)
        elif ext == '.md':
            return UnstructuredMarkdownLoader(file_path)
        else:
            logger.warning(f"Unsupported file type: {ext}. Skipping {file_path}")
            return None
    
    def load_documents(self) -> bool:
        """
        Load documents from upload directory and create vector embeddings.
        Returns True if successful, False otherwise.
        """
        print(f"📂 Loading documents from: {self.docs_dir}")
        
        # Check if directory exists
        if not os.path.exists(self.docs_dir):
            logger.error(f"Document directory does not exist: {self.docs_dir}")
            return False
        
        # Get all document files
        all_files = []
        for ext in ['.pdf', '.docx', '.txt', '.md']:
            all_files.extend(glob.glob(os.path.join(self.docs_dir, f"**/*{ext}"), recursive=True))
        
        if not all_files:
            print(f"📭 No supported documents found in {self.docs_dir}")
            return False
        
        print(f"📚 Found {len(all_files)} documents to process")
        
        # Process each document
        documents = []
        for file_path in all_files:
            try:
                loader = self._get_document_loader(file_path)
                if loader:
                    print(f"📖 Loading {os.path.basename(file_path)}")
                    docs = loader.load()
                    documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        if not documents:
            logger.error("Failed to load any documents")
            return False
        
        # Split documents into chunks
        print("🔪 Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"📝 Created {len(chunks)} document chunks")
        
        # Create vector store
        print("🔗 Creating FAISS vector database...")
        try:
            self.vector_db = FAISS.from_documents(chunks, self.embeddings)
            # Save for future use
            self.vector_db.save_local(self.cache_dir)
            print(f"💾 Vector database created and saved to {self.cache_dir}")
            return True
        except Exception as e:
            logger.error(f"Error creating vector database: {e}")
            return False
    
    def load_existing_db(self) -> bool:
        """
        Load existing vector database if available.
        Returns True if successful, False otherwise.
        """
        if os.path.exists(self.cache_dir):
            print(f"🔄 Loading existing vector database from {self.cache_dir}")
            try:
                self.vector_db = FAISS.load_local(
                    self.cache_dir, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                print("✅ Vector database loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading vector database: {e}")
                print(f"⚠️ Failed to load existing database: {e}")
                return False
        else:
            return False
    
    def search_documents(self, query: str, k: Optional[int] = None) -> List[str]:
        """
        Search for relevant document chunks based on query.
        
        Args:
            query: The user query to search for
            k: Number of results to return (uses config default if None)
            
        Returns:
            List of relevant document chunks
        """
        if not self.vector_db:
            print("⚠️ Vector database not initialized. Please load documents first.")
            return []
        
        k = k or self.top_k
        
        try:
            print(f"🔍 Searching for relevant documents (top {k})...")
            results = self.vector_db.similarity_search(query, k=k)
            documents = [doc.page_content for doc in results]
            
            print(f"📊 Found {len(documents)} relevant documents:")
            for i, doc in enumerate(documents):
                print(f"   📄 Document {i+1}: {doc[:100]}...")
            
            return documents
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            print(f"❌ Search failed: {e}")
            return []
    
    def search_documents_with_scores(self, query: str, k: Optional[int] = None) -> List[tuple]:
        """
        Search for relevant documents with similarity scores.
        
        Args:
            query: The user query to search for
            k: Number of results to return
            
        Returns:
            List of tuples (document, score)
        """
        if not self.vector_db:
            print("⚠️ Vector database not initialized")
            return []
        
        k = k or self.top_k
        
        try:
            print(f"🔍 Searching with scores (top {k})...")
            results = self.vector_db.similarity_search_with_score(query, k=k)
            
            print(f"📊 Found {len(results)} documents with scores:")
            for i, (doc, score) in enumerate(results):
                print(f"   📄 Document {i+1} (score: {score:.3f}): {doc.page_content[:80]}...")
            
            return [(doc.page_content, score) for doc, score in results]
        except Exception as e:
            logger.error(f"Error searching documents with scores: {e}")
            return []
    
    def retrieve(self, query_text: str, top_k: Optional[int] = None) -> List[str]:
        """
        Retrieve relevant documents from the database.
        
        Args:
            query_text: The user's query to search for
            top_k: Maximum number of results to retrieve
        
        Returns:
            List of relevant text chunks
        """
        print(f"🚀 Starting document retrieval...")
        print(f"❓ Query: {query_text[:80]}...")
        
        # Search for documents
        documents = self.search_documents(query_text, top_k)
        
        if not documents:
            print("📭 No relevant documents found")
            # Try to reload documents if none found
            if self.load_documents():
                print("🔄 Retrying search after loading documents...")
                documents = self.search_documents(query_text, top_k)
        
        print(f"✅ Retrieved {len(documents)} documents\n")
        return documents
    
    def format_context(self, docs: List[str]) -> str:
        """
        Format retrieved documents into a single context string.
        
        Args:
            docs: List of document text chunks
        
        Returns:
            Formatted context string with all documents combined
        """
        if not docs:
            print("📭 No documents to format")
            return ""
        
        print(f"📝 Formatting {len(docs)} documents into context...")
        
        # Join documents with clear separators
        context = "\n\n---DOCUMENT SEPARATOR---\n\n".join(docs)
        
        # Display context info
        context_length = len(context)
        print(f"✅ Context created: {context_length} characters, {len(docs)} chunks")
        
        return context
    
    def retrieve_and_format(self, query_text: str, top_k: Optional[int] = None) -> str:
        """
        Convenience method to retrieve and format in one step.
        
        Args:
            query_text: The user's query
            top_k: Maximum number of results
        
        Returns:
            Formatted context string ready for LLM
        """
        print("🎯 Retrieve and format pipeline starting...")
        
        # Retrieve relevant documents
        relevant_docs = self.retrieve(query_text, top_k)
        
        # Format into context string
        context = self.format_context(relevant_docs)
        
        print("🏁 Pipeline complete!\n")
        return context
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        if not self.vector_db:
            return {"status": "not_initialized", "document_count": 0}
        
        try:
            # Get document count from FAISS index
            doc_count = self.vector_db.index.ntotal if hasattr(self.vector_db, 'index') else 0
            
            return {
                "status": "ready",
                "document_count": doc_count,
                "cache_dir": self.cache_dir,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def reset_database(self) -> bool:
        """Reset the vector database by deleting cache and reinitializing."""
        try:
            print("🗑️ Resetting vector database...")
            
            # Clear current database
            self.vector_db = None
            
            # Remove cache directory
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
            
            # Recreate directory
            os.makedirs(self.cache_dir, exist_ok=True)
            
            print("✅ Vector database reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            return False