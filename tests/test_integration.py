"""
Integration tests for AI Study Assistant API endpoints.
Tests the complete API workflow including health checks, chat, summarization, and document management.
"""

import pytest
from fastapi.testclient import TestClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def client():
    """Create test client."""
    from api.main import app
    return TestClient(app)


def test_health_check(client):
    """Test API health endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'
    logger.info('✓ Health check passed')


def test_chat_endpoint(client):
    """Test chat endpoint."""
    response = client.post('/chat?query=What is AI?&top_k=5')
    assert response.status_code == 200
    data = response.json()
    assert 'answer' in data
    assert 'sources' in data
    logger.info('✓ Chat endpoint passed')


def test_summarize_endpoint(client):
    """Test summarize endpoint."""
    text = 'This is a test document. It contains multiple sentences. Testing summarization.'
    response = client.post(f'/summarize?text={text}&summary_type=extractive')
    assert response.status_code == 200
    data = response.json()
    assert 'summary' in data
    logger.info('✓ Summarize endpoint passed')


def test_extract_keywords_endpoint(client):
    """Test keyword extraction endpoint."""
    text = 'Machine learning and artificial intelligence are important technologies in the modern world.'
    response = client.post(f'/extract-keywords?text={text}&top_n=5')
    assert response.status_code == 200
    data = response.json()
    assert 'keywords' in data
    logger.info('✓ Keyword extraction passed')


def test_documents_endpoint(client):
    """Test documents listing endpoint."""
    response = client.get('/documents')
    assert response.status_code == 200
    data = response.json()
    assert 'total_documents' in data
    logger.info('✓ Documents endpoint passed')


# Run with: pytest tests/test_integration.py -v
