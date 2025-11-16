from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import Config
from src.inference.chromadb_manager import ChromaDBManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGRetriever:
    """Retrieval-Augmented Generation retriever with advanced features."""
    
    def __init__(
        self,
        vector_store_manager: ChromaDBManager,
        embedding_model_name: Optional[str] = None,
        reranker_model: Optional[str] = None,
        config: Optional[Config] = None
    ):
        """Initialize RAG retriever."""
        self.config = config or Config()
        self.vector_store = vector_store_manager
        
        # Load embedding model (use same as vector store for consistency)
        if embedding_model_name is None:
            embedding_model_name = self.config.EMBEDDING_MODEL_NAME
        
        logger.info(f'Initializing RAG retriever with embedding model: {embedding_model_name}')
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Optional: Load reranker model for better results
        self.reranker = None
        if reranker_model:
            logger.info(f'Loading reranker model: {reranker_model}')
            self.reranker = CrossEncoder(reranker_model)
        
        logger.info('RAG retriever initialized')
    
    def retrieve_documents(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """Retrieve relevant documents with filtering and reranking."""
        if top_k is None:
            top_k = self.config.TOP_K_RETRIEVAL
        if similarity_threshold is None:
            similarity_threshold = self.config.SIMILARITY_THRESHOLD
        
        logger.info(f'Retrieving documents for query: "{query_text[:50]}..."')
        logger.info(f'Parameters: top_k={top_k}, threshold={similarity_threshold}')
        
        # Retrieve more candidates for reranking
        retrieval_k = top_k * 2 if self.reranker else top_k
        
        # Query vector store
        retrieved = self.vector_store.query(
            query_text,
            n_results=retrieval_k,
            where=metadata_filter
        )
        
        # Filter by similarity threshold
        filtered = [
            {
                'text': doc['document'],
                'metadata': doc['metadata'],
                'score': doc['similarity'],
                'id': doc['id']
            }
            for doc in retrieved
            if doc['similarity'] >= similarity_threshold
        ]
        
        logger.info(f'Filtered to {len(filtered)} documents above threshold')
        
        # Rerank if reranker is available
        if self.reranker and filtered:
            filtered = self._rerank_documents(query_text, filtered)
        
        # Return top k
        result = filtered[:top_k]
        logger.info(f'Returning {len(result)} documents')
        return result
    
    def _rerank_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Rerank documents using cross-encoder."""
        logger.info('Reranking documents with cross-encoder...')
        
        # Prepare pairs for reranking
        pairs = [(query, doc['text']) for doc in documents]
        
        # Get reranking scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Update scores
        for doc, score in zip(documents, rerank_scores):
            doc['rerank_score'] = float(score)
            doc['original_score'] = doc['score']
            doc['score'] = float(score)  # Use rerank score as primary
        
        # Sort by rerank score
        documents.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info('Reranking completed')
        return documents
    
    def expand_query(self, query_text: str, num_expansions: int = 3) -> List[str]:
        """Generate query variations for multi-query retrieval."""
        logger.info(f'Expanding query into {num_expansions} variations')
        
        variations = [query_text]
        words = query_text.split()
        
        if len(words) > 3:
            # Variation 1: First half of query
            mid_point = len(words) // 2
            variations.append(' '.join(words[:mid_point]))
            
            # Variation 2: Second half of query
            variations.append(' '.join(words[mid_point:]))
            
            # Variation 3: Remove stop words (simple version)
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'why'}
            filtered_words = [w for w in words if w.lower() not in stop_words]
            if filtered_words and filtered_words != words:
                variations.append(' '.join(filtered_words))
        
        # Return unique variations, limited by num_expansions
        unique_variations = []
        seen = set()
        for var in variations:
            if var not in seen:
                unique_variations.append(var)
                seen.add(var)
                if len(unique_variations) >= num_expansions:
                    break
        
        logger.info(f'Generated {len(unique_variations)} query variations')
        return unique_variations
    
    def multi_query_retrieve(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        num_expansions: int = 3
    ) -> List[Dict]:
        """Retrieve using multiple query variations with score fusion."""
        if top_k is None:
            top_k = self.config.TOP_K_RETRIEVAL
        
        logger.info('Performing multi-query retrieval')
        
        # Generate query variations
        expanded_queries = self.expand_query(query_text, num_expansions)
        
        # Track documents and their scores
        doc_scores = {}
        doc_data = {}
        
        # Retrieve for each query variation
        for i, expanded_query in enumerate(expanded_queries):
            logger.info(f'Query variation {i+1}: "{expanded_query}"')
            docs = self.retrieve_documents(expanded_query, top_k=top_k * 2)
            
            # Weight scores (original query gets more weight)
            weight = 1.0 if i == 0 else 0.7
            
            for doc in docs:
                doc_id = doc['id']
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                    doc_data[doc_id] = doc
                
                # Accumulate weighted scores
                doc_scores[doc_id] += doc['score'] * weight
        
        # Sort by accumulated score
        results = []
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            doc = doc_data[doc_id].copy()
            doc['score'] = score
            doc['retrieval_method'] = 'multi_query'
            results.append(doc)
        
        final_results = results[:top_k]
        logger.info(f'Multi-query retrieval returned {len(final_results)} documents')
        return final_results
    
    def hybrid_retrieve(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict] = None,
        use_multi_query: bool = True
    ) -> List[Dict]:
        """Hybrid retrieval combining multiple strategies."""
        if top_k is None:
            top_k = self.config.TOP_K_RETRIEVAL
        
        logger.info('Performing hybrid retrieval')
        
        if use_multi_query:
            results = self.multi_query_retrieve(query_text, top_k=top_k)
        else:
            results = self.retrieve_documents(
                query_text,
                top_k=top_k,
                metadata_filter=metadata_filter
            )
        
        return results
    
    def retrieve_with_context(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        context_window: int = 1
    ) -> List[Dict]:
        """Retrieve documents with surrounding context."""
        if top_k is None:
            top_k = self.config.TOP_K_RETRIEVAL
        
        logger.info('Retrieving documents with context window')
        
        # Get initial results
        results = self.retrieve_documents(query_text, top_k=top_k)
        
        # For each result, try to get surrounding context
        # This is a simple implementation - can be enhanced based on document structure
        enhanced_results = []
        for result in results:
            enhanced_result = result.copy()
            
            # Try to get context from metadata if available
            if 'chunk_id' in result['metadata']:
                chunk_id = result['metadata']['chunk_id']
                # Could retrieve adjacent chunks here
                enhanced_result['has_context'] = True
            else:
                enhanced_result['has_context'] = False
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def get_retrieval_stats(self) -> Dict:
        """Get retrieval statistics."""
        vector_stats = self.vector_store.get_collection_stats()
        
        stats = {
            **vector_stats,
            'embedding_model': self.config.EMBEDDING_MODEL_NAME,
            'has_reranker': self.reranker is not None,
            'default_top_k': self.config.TOP_K_RETRIEVAL,
            'similarity_threshold': self.config.SIMILARITY_THRESHOLD
        }
        
        return stats

# Usage:
# # Initialize with vector store
# from src.inference.chromadb_manager import ChromaDBManager
# vector_store = ChromaDBManager(collection_name='study_materials')
# retriever = RAGRetriever(vector_store)
#
# # Simple retrieval
# docs = retriever.retrieve_documents("What is machine learning?", top_k=5)
#
# # Multi-query retrieval
# docs = retriever.multi_query_retrieve("Explain neural networks", top_k=5)
#
# # Hybrid retrieval with metadata filter
# docs = retriever.hybrid_retrieve(
#     "Deep learning concepts",
#     top_k=5,
#     metadata_filter={'topic': 'AI'}
# )
