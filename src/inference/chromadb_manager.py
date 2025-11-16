import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
import numpy as np

# Import project config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaDBManager:
    """Manager for ChromaDB vector database with persistent storage."""
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = 'documents',
        embedding_model_name: Optional[str] = None,
        config: Optional[Config] = None
    ):
        """Initialize ChromaDB manager with persistence."""
        self.config = config or Config()
        
        # Use config values if not provided
        if persist_directory is None:
            persist_directory = self.config.PERSIST_DIRECTORY
        if embedding_model_name is None:
            embedding_model_name = self.config.EMBEDDING_MODEL_NAME
        
        # Ensure persist directory exists
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f'Initializing ChromaDB at {persist_directory}')
        
        # Initialize ChromaDB with persistent client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Initialize embedding model
        logger.info(f'Loading embedding model: {embedding_model_name}')
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        logger.info(f'ChromaDB initialized with embedding dimension: {self.embedding_dimension}')
        logger.info(f'Collection "{collection_name}" has {self.collection.count()} documents')
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ):
        """Add documents with embeddings to collection."""
        if not documents:
            logger.warning('No documents to add')
            return
        
        num_docs = len(documents)
        logger.info(f'Adding {num_docs} documents to collection "{self.collection_name}"...')
        
        # Generate IDs if not provided
        if ids is None:
            existing_count = self.collection.count()
            ids = [f'doc_{existing_count + i}' for i in range(num_docs)]
        
        # Generate empty metadata if not provided
        if metadatas is None:
            metadatas = [{'source': 'unknown'} for _ in range(num_docs)]
        
        # Validate inputs
        if len(metadatas) != num_docs:
            raise ValueError(f"Metadata length ({len(metadatas)}) must match documents length ({num_docs})")
        if len(ids) != num_docs:
            raise ValueError(f"IDs length ({len(ids)}) must match documents length ({num_docs})")
        
        # Generate embeddings
        logger.info('Generating embeddings...')
        embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f'Successfully added {num_docs} documents. Total: {self.collection.count()}')
    
    def query(
        self,
        query_text: str,
        n_results: int = None,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> List[Dict]:
        """Query similar documents from collection."""
        if n_results is None:
            n_results = self.config.TOP_K_RETRIEVAL
        
        logger.info(f'Querying collection with: "{query_text[:50]}..." (top {n_results})')
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text], convert_to_numpy=True)
        
        # Query collection
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        
        # Format results
        retrieved = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 0
                doc_id = results['ids'][0][i] if results['ids'] else None
                
                # Convert distance to similarity score (for cosine: similarity = 1 - distance)
                similarity = 1 - distance
                
                retrieved.append({
                    'id': doc_id,
                    'document': doc,
                    'metadata': metadata,
                    'similarity': similarity,
                    'distance': distance
                })
        
        logger.info(f'Retrieved {len(retrieved)} documents')
        return retrieved
    
    def query_with_filter(
        self,
        query_text: str,
        filter_dict: Dict,
        n_results: int = None
    ) -> List[Dict]:
        """Query with metadata filtering."""
        return self.query(query_text, n_results=n_results, where=filter_dict)
    
    def get_by_ids(self, ids: List[str]) -> List[Dict]:
        """Retrieve documents by their IDs."""
        logger.info(f'Retrieving {len(ids)} documents by ID')
        
        results = self.collection.get(ids=ids)
        
        retrieved = []
        for i in range(len(results['ids'])):
            retrieved.append({
                'id': results['ids'][i],
                'document': results['documents'][i],
                'metadata': results['metadatas'][i] if results['metadatas'] else {}
            })
        
        return retrieved
    
    def update_documents(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None
    ):
        """Update existing documents."""
        logger.info(f'Updating {len(ids)} documents')
        
        update_kwargs = {'ids': ids}
        
        if documents:
            embeddings = self.embedding_model.encode(documents, convert_to_numpy=True)
            update_kwargs['embeddings'] = embeddings.tolist()
            update_kwargs['documents'] = documents
        
        if metadatas:
            update_kwargs['metadatas'] = metadatas
        
        self.collection.update(**update_kwargs)
        logger.info('Documents updated successfully')
    
    def delete_documents(self, ids: List[str]):
        """Delete documents by IDs."""
        logger.info(f'Deleting {len(ids)} documents')
        self.collection.delete(ids=ids)
        logger.info('Documents deleted successfully')
    
    def delete_collection(self):
        """Delete and recreate collection."""
        logger.warning(f'Deleting collection "{self.collection_name}"...')
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info('Collection recreated')
    
    def clear_collection(self):
        """Remove all documents from collection without deleting it."""
        logger.warning('Clearing all documents from collection...')
        count = self.collection.count()
        if count > 0:
            # Get all IDs and delete them
            all_data = self.collection.get()
            if all_data['ids']:
                self.collection.delete(ids=all_data['ids'])
        logger.info(f'Cleared {count} documents from collection')
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        count = self.collection.count()
        stats = {
            'collection_name': self.collection_name,
            'document_count': count,
            'embedding_dimension': self.embedding_dimension,
            'embedding_model': self.config.EMBEDDING_MODEL_NAME
        }
        logger.info(f'Collection stats: {stats}')
        return stats
    
    def peek(self, limit: int = 10) -> Dict:
        """Peek at the first few documents in collection."""
        logger.info(f'Peeking at first {limit} documents')
        return self.collection.peek(limit=limit)
    
    def search_by_metadata(self, metadata_filter: Dict, limit: int = 10) -> List[Dict]:
        """Search documents by metadata only."""
        logger.info(f'Searching by metadata: {metadata_filter}')
        results = self.collection.get(where=metadata_filter, limit=limit)
        
        retrieved = []
        for i in range(len(results['ids'])):
            retrieved.append({
                'id': results['ids'][i],
                'document': results['documents'][i],
                'metadata': results['metadatas'][i] if results['metadatas'] else {}
            })
        
        return retrieved

# Usage:
# # Initialize with defaults from config
# manager = ChromaDBManager(collection_name='study_materials')
#
# # Add documents
# docs = ["AI is...", "Machine learning is..."]
# metadatas = [{'topic': 'AI'}, {'topic': 'ML'}]
# manager.add_documents(docs, metadatas)
#
# # Query similar documents
# results = manager.query('What is artificial intelligence?', n_results=5)
# for result in results:
#     print(f"Similarity: {result['similarity']:.3f}")
#     print(f"Document: {result['document'][:100]}...")
