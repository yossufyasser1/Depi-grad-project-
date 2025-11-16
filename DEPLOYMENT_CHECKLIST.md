# Deployment Checklist

## Pre-Deployment Verification

### Code Quality
- [ ] All tests passing: `pytest tests/ -v`
- [ ] Integration tests passed: `pytest tests/test_integration.py -v`
- [ ] Unit tests passed: `pytest tests/test_unit.py -v`
- [ ] Code formatted: `black src/ api/ tests/`
- [ ] Linting passed: `flake8 src/ api/ tests/`
- [ ] Type checking: `mypy src/ api/` (optional)

### Configuration
- [ ] `requirements.txt` updated and version pinned
- [ ] `.env` file configured with production values
- [ ] `.env.example` created for documentation
- [ ] `.gitignore` includes sensitive files
- [ ] Config paths verified in `src/config.py`

### Docker & Containers
- [ ] Docker image builds: `docker build -t study-assistant .`
- [ ] Docker image size reasonable: `docker images study-assistant`
- [ ] Docker compose runs: `docker-compose up -d`
- [ ] Health check passes: `curl http://localhost:8000/health`
- [ ] Container logs clean: `docker logs ai-study-assistant`

### Data & Models
- [ ] All required models downloaded locally
- [ ] Model cache directory exists: `./models`
- [ ] Vector database initialized: `./chroma_db`
- [ ] Test datasets downloaded: `python src/preprocessing/download_datasets.py`
- [ ] MLflow tracking configured: `./mlruns`

### Security
- [ ] Non-root user in Docker container
- [ ] Sensitive data not in Git repository
- [ ] Environment variables used for secrets
- [ ] CORS properly configured
- [ ] API rate limiting considered

## Deployment Steps

### 1. Build Docker Image
```bash
docker build -t study-assistant:latest .
```

### 2. Test Locally
```bash
# Run container
docker run -p 8000:8000 study-assistant:latest

# Or use docker-compose
docker-compose up -d
```

### 3. Health Checks
```bash
# Check API health
curl http://localhost:8000/health

# Check API documentation
# Open: http://localhost:8000/docs

# Test endpoints
curl -X POST "http://localhost:8000/chat?query=What%20is%20AI&top_k=5"
```

### 4. Monitoring
```bash
# View logs
docker logs -f ai-study-assistant

# Check resource usage
docker stats ai-study-assistant

# Monitor system resources
python src/performance_profiler.py
```

## Performance Targets

### Response Times
- API health check: < 100ms
- Document retrieval: < 500ms
- Question answering: < 2 seconds
- Summarization (extractive): < 1 second
- Summarization (abstractive): < 5 seconds
- Keyword extraction: < 500ms

### Resource Usage
- Memory usage: < 4GB
- CPU usage: < 80%
- Disk I/O: Minimal
- Error rate: < 1%

### Scalability
- Concurrent requests: > 10
- Documents in vector DB: > 10,000
- Query throughput: > 100/min

## Rollback Procedure

### Tag Previous Version
```bash
# Before deploying new version
docker tag study-assistant:latest study-assistant:previous
```

### Revert to Previous Version
```bash
# Stop current container
docker-compose down

# Run previous version
docker run -p 8000:8000 study-assistant:previous
```

### Emergency Rollback
```bash
# Quick rollback
docker stop ai-study-assistant
docker rm ai-study-assistant
docker run -d -p 8000:8000 --name ai-study-assistant study-assistant:previous
```

## Post-Deployment

### Verification
- [ ] Health endpoint responding: `/health`
- [ ] API documentation accessible: `/docs`
- [ ] Chat endpoint working: `/chat`
- [ ] Upload endpoint working: `/upload`
- [ ] Summarization working: `/summarize`
- [ ] All endpoints return expected responses

### Monitoring
- [ ] Set up application logging
- [ ] Configure error alerting
- [ ] Monitor resource usage
- [ ] Track API metrics
- [ ] Review MLflow experiments

### Maintenance
- [ ] Document deployment date
- [ ] Update version tags
- [ ] Backup vector database
- [ ] Archive old logs
- [ ] Clean up unused Docker images

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker logs ai-study-assistant

# Verify configuration
docker inspect ai-study-assistant

# Check ports
netstat -an | findstr 8000
```

### High Memory Usage
```bash
# Monitor memory
docker stats ai-study-assistant

# Restart container
docker-compose restart

# Reduce model size or enable quantization
```

### Slow Response Times
```bash
# Run performance profiler
python src/performance_profiler.py

# Check system resources
python -c "from src.performance_profiler import system_resources; system_resources()"

# Enable GPU if available
```

## Notes

- Always test in staging environment before production
- Keep at least 2 previous Docker image versions
- Document any configuration changes
- Review security best practices regularly
- Update dependencies monthly
