"""
Unit tests for AI Study Assistant core components.
Tests individual modules including preprocessing, summarization, keyword extraction, and vector store.
"""

import pytest
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_text_preprocessor():
    """Test text preprocessing."""
    from src.preprocessing.text_preprocessor import TextPreprocessor
    
    preprocessor = TextPreprocessor()
    text = 'Machine learning is amazing!'
    
    result = preprocessor.preprocess_full(text)
    
    assert 'sentences' in result
    assert 'tokens' in result
    assert len(result['tokens']) > 0
    logger.info('✓ Text preprocessor test passed')


def test_textrank_summarizer():
    """Test TextRank summarizer."""
    from src.training.textrank_summarizer import TextRankSummarizer
    
    summarizer = TextRankSummarizer(num_sentences=2)
    text = '''Machine learning is fascinating. It powers many applications. 
    Neural networks are complex. Deep learning is powerful.'''
    
    summary = summarizer.summarize(text)
    
    assert len(summary) < len(text)
    assert len(summary) > 0
    logger.info('✓ TextRank test passed')


def test_keyword_extractor():
    """Test keyword extraction."""
    from src.training.keyword_extractor import KeywordExtractor
    
    extractor = KeywordExtractor()
    text = 'Python programming language is widely used in data science and machine learning.'
    
    keywords = extractor.extract_keywords(text, top_n=5)
    
    assert len(keywords) > 0
    assert all(isinstance(k, tuple) and len(k) == 2 for k in keywords)
    logger.info('✓ Keyword extractor test passed')


def test_chromadb_manager():
    """Test ChromaDB manager."""
    from src.inference.chromadb_manager import ChromaDBManager
    
    # Use a temporary test database
    test_db_path = './test_chroma_db'
    
    try:
        manager = ChromaDBManager(test_db_path)
        
        docs = ['This is a test document', 'Another test document']
        metadata = [{'source': 'test1'}, {'source': 'test2'}]
        ids = ['doc1', 'doc2']
        
        manager.add_documents(docs, metadata, ids)
        
        stats = manager.get_collection_stats()
        assert stats['document_count'] == 2
        logger.info('✓ ChromaDB manager test passed')
    
    finally:
        # Clean up test database
        if os.path.exists(test_db_path):
            shutil.rmtree(test_db_path)


def test_config():
    """Test configuration loading."""
    from src.config import Config
    
    config = Config()
    
    # Check that essential paths are set
    assert config.RAW_DATA_PATH is not None
    assert config.PROCESSED_DATA_PATH is not None
    assert config.MODEL_SAVE_PATH is not None
    
    # Check model names
    assert config.EMBEDDING_MODEL_NAME is not None
    assert config.LLM_MODEL_NAME is not None
    
    logger.info('✓ Config test passed')


def test_evaluation_metrics():
    """Test evaluation metrics computation."""
    from src.evaluation_metrics import EvaluationMetrics
    
    metrics = EvaluationMetrics()
    
    # Test ROUGE
    reference = "The cat sat on the mat"
    hypothesis = "The cat is on the mat"
    rouge_scores = metrics.compute_rouge(reference, hypothesis)
    assert 'rouge1' in rouge_scores
    assert 'rouge2' in rouge_scores
    assert 'rougeL' in rouge_scores
    
    # Test BLEU
    bleu_score = metrics.compute_bleu(reference, hypothesis)
    assert 0.0 <= bleu_score <= 1.0
    
    logger.info('✓ Evaluation metrics test passed')


def test_pdf_processor():
    """Test PDF processor initialization."""
    from src.preprocessing.pdf_processor import PDFProcessor
    
    processor = PDFProcessor()
    
    # Test text chunking
    text = "This is a test. " * 100  # Long text
    chunks = processor.chunk_text(text, chunk_size=50)
    
    assert len(chunks) > 1
    assert all(len(chunk) <= 50 for chunk in chunks)
    
    logger.info('✓ PDF processor test passed')


# Run with: pytest tests/test_unit.py -v
