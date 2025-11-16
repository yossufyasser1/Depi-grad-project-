"""
Performance profiling utilities for AI Study Assistant.
Provides decorators and functions for profiling execution time, memory usage, and model performance.
"""

import time
import functools
from typing import Callable, Any, Dict
import psutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def profile_execution(func: Callable) -> Callable:
    """
    Decorator to profile function execution time and memory usage.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function with profiling
        
    Example:
        @profile_execution
        def my_function():
            # Your code here
            pass
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Get initial resources
        start_time = time.time()
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Get final resources
        end_time = time.time()
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        elapsed_time = end_time - start_time
        mem_used = mem_after - mem_before
        
        logger.info(
            f'{func.__name__}: {elapsed_time:.2f}s, '
            f'Memory: {mem_used:+.2f}MB (Before: {mem_before:.2f}MB, After: {mem_after:.2f}MB)'
        )
        
        return result
    
    return wrapper


@profile_execution
def test_retrieval_performance() -> Dict[str, Any]:
    """
    Test performance of vector store retrieval.
    
    Returns:
        Query results from ChromaDB
    """
    from src.inference.chromadb_manager import ChromaDBManager
    
    manager = ChromaDBManager()
    
    # Simulate query
    results = manager.query('test query', n_results=5)
    
    # Handle results (list of dicts from ChromaDB)
    num_docs = len(results) if results else 0
    logger.info(f'Retrieved {num_docs} documents')
    
    return {'documents': results, 'count': num_docs}


def benchmark_summarization() -> Dict[str, float]:
    """
    Benchmark different summarization approaches.
    
    Returns:
        Dictionary with timing and compression metrics
    """
    from src.training.textrank_summarizer import TextRankSummarizer
    from src.training.bart_summarizer import BARTSummarizer
    
    test_text = '''Machine learning is fascinating. It powers many applications. 
    Neural networks are complex structures. Deep learning is powerful. 
    Artificial intelligence transforms industries. Data science is essential. 
    Python is widely used for ML. TensorFlow and PyTorch are popular frameworks.
    Natural language processing enables text understanding. Computer vision analyzes images.''' * 10
    
    results = {}
    
    # Test TextRank (extractive)
    logger.info('Benchmarking TextRank...')
    try:
        summarizer = TextRankSummarizer(num_sentences=3)
        
        start = time.time()
        summary = summarizer.summarize(test_text)
        end = time.time()
        
        textrank_time = end - start
        textrank_compression = len(summary) / len(test_text) * 100
        
        results['textrank'] = {
            'time': textrank_time,
            'compression_ratio': textrank_compression
        }
        
        logger.info(f'TextRank: {textrank_time:.3f}s, Compression: {textrank_compression:.1f}%')
    except Exception as e:
        logger.error(f'TextRank benchmark failed: {e}')
    
    # Test BART (abstractive) - requires model
    logger.info('Benchmarking BART...')
    try:
        bart = BARTSummarizer()
        bart.load_model()
        
        start = time.time()
        summary = bart.generate_summary(test_text[:1024])  # Limit input size
        end = time.time()
        
        bart_time = end - start
        bart_compression = len(summary) / len(test_text[:1024]) * 100
        
        results['bart'] = {
            'time': bart_time,
            'compression_ratio': bart_compression
        }
        
        logger.info(f'BART: {bart_time:.3f}s, Compression: {bart_compression:.1f}%')
    except Exception as e:
        logger.error(f'BART benchmark failed: {e}')
    
    return results


def benchmark_keyword_extraction() -> Dict[str, float]:
    """
    Benchmark keyword extraction performance.
    
    Returns:
        Dictionary with timing metrics
    """
    from src.training.keyword_extractor import KeywordExtractor
    
    test_text = '''Python programming language is widely used in data science and machine learning.
    Natural language processing and computer vision are key areas of artificial intelligence.
    Deep learning models like neural networks require significant computational resources.
    TensorFlow and PyTorch are popular frameworks for building machine learning models.''' * 5
    
    extractor = KeywordExtractor()
    
    start = time.time()
    keywords = extractor.extract_keywords(test_text, top_n=10)
    end = time.time()
    
    elapsed = end - start
    
    logger.info(f'Keyword Extraction: {elapsed:.3f}s')
    logger.info(f'Extracted {len(keywords)} keywords: {[k[0] for k in keywords[:5]]}...')
    
    return {
        'time': elapsed,
        'num_keywords': len(keywords)
    }


def benchmark_rag_pipeline() -> Dict[str, float]:
    """
    Benchmark complete RAG pipeline performance.
    
    Returns:
        Dictionary with timing metrics for each stage
    """
    from src.inference.rag_retriever import RAGRetriever
    from src.inference.llm_reader import LLMReader
    
    query = "What is machine learning and how does it work?"
    
    results = {}
    
    # Retrieval stage
    logger.info('Benchmarking retrieval...')
    try:
        from src.inference.chromadb_manager import ChromaDBManager
        
        vector_store = ChromaDBManager()
        retriever = RAGRetriever(vector_store_manager=vector_store)
        
        start = time.time()
        docs = retriever.retrieve_documents(query, top_k=5)
        end = time.time()
        
        retrieval_time = end - start
        results['retrieval'] = retrieval_time
        
        logger.info(f'Retrieval: {retrieval_time:.3f}s, Retrieved {len(docs)} documents')
    except Exception as e:
        logger.error(f'Retrieval benchmark failed: {e}')
        docs = []
    
    # Generation stage
    logger.info('Benchmarking generation...')
    try:
        reader = LLMReader()
        
        context = "\n\n".join([doc['text'] for doc in docs[:3]]) if docs else "Sample context text."
        
        start = time.time()
        answer = reader.generate_answer(query, context)
        end = time.time()
        
        generation_time = end - start
        results['generation'] = generation_time
        
        logger.info(f'Generation: {generation_time:.3f}s, Answer length: {len(answer)} chars')
    except Exception as e:
        logger.error(f'Generation benchmark failed: {e}')
    
    # Total time
    if 'retrieval' in results and 'generation' in results:
        results['total'] = results['retrieval'] + results['generation']
        logger.info(f'Total RAG Pipeline: {results["total"]:.3f}s')
    
    return results


def system_resources() -> Dict[str, float]:
    """
    Get current system resource usage.
    
    Returns:
        Dictionary with CPU, memory, and disk usage
    """
    process = psutil.Process()
    
    resources = {
        'cpu_percent': process.cpu_percent(interval=1.0),
        'memory_mb': process.memory_info().rss / 1024 / 1024,
        'memory_percent': process.memory_percent(),
        'num_threads': process.num_threads(),
        'system_cpu_percent': psutil.cpu_percent(interval=1.0),
        'system_memory_percent': psutil.virtual_memory().percent,
        'system_disk_percent': psutil.disk_usage('/').percent
    }
    
    logger.info('System Resources:')
    logger.info(f'  CPU: {resources["cpu_percent"]:.1f}% (System: {resources["system_cpu_percent"]:.1f}%)')
    logger.info(f'  Memory: {resources["memory_mb"]:.1f}MB ({resources["memory_percent"]:.1f}%)')
    logger.info(f'  System Memory: {resources["system_memory_percent"]:.1f}%')
    logger.info(f'  Threads: {resources["num_threads"]}')
    
    return resources


def run_full_benchmark() -> Dict[str, Any]:
    """
    Run comprehensive benchmark of all components.
    
    Returns:
        Dictionary with all benchmark results
    """
    logger.info('=' * 60)
    logger.info('Running Full Performance Benchmark')
    logger.info('=' * 60)
    
    results = {}
    
    # System resources
    logger.info('\n[1/5] System Resources')
    results['system'] = system_resources()
    
    # Summarization
    logger.info('\n[2/5] Summarization Benchmark')
    results['summarization'] = benchmark_summarization()
    
    # Keyword extraction
    logger.info('\n[3/5] Keyword Extraction Benchmark')
    results['keyword_extraction'] = benchmark_keyword_extraction()
    
    # Retrieval
    logger.info('\n[4/5] Retrieval Benchmark')
    try:
        results['retrieval'] = test_retrieval_performance()
    except Exception as e:
        logger.error(f'Retrieval benchmark failed: {e}')
    
    # RAG pipeline
    logger.info('\n[5/5] RAG Pipeline Benchmark')
    results['rag_pipeline'] = benchmark_rag_pipeline()
    
    logger.info('\n' + '=' * 60)
    logger.info('Benchmark Complete')
    logger.info('=' * 60)
    
    return results


# Usage examples:
if __name__ == '__main__':
    # Run individual benchmarks
    # test_retrieval_performance()
    # benchmark_summarization()
    # benchmark_keyword_extraction()
    # benchmark_rag_pipeline()
    # system_resources()
    
    # Run full benchmark suite
    results = run_full_benchmark()
