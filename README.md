# AI-Powered Study Assistant 📚

A Mini NotebookLM clone for intelligent document processing, question answering, and knowledge management using Retrieval-Augmented Generation (RAG).

## 🌟 Features

### Core Capabilities
- **PDF Processing**: Upload and extract text from PDFs with OCR support
- **Text Summarization**: 
  - Extractive summarization using TextRank algorithm
  - Abstractive summarization using BART
- **Named Entity Recognition**: Extract entities using fine-tuned BERT
- **Keyword Extraction**: Automatic keyword identification using RAKE
- **Semantic Search**: ChromaDB vector database for similarity search
- **Question Answering**: RAG-powered Q&A using DistilBERT
- **LLM Generation**: Generate answers with GPT-2 or custom models

### Technical Features
- **FastAPI Backend**: RESTful API with automatic documentation
- **Fine-tuning Support**: T5 with LoRA for efficient adaptation
- **Experiment Tracking**: MLflow integration for model versioning
- **Performance Profiling**: Built-in benchmarking and monitoring
- **Docker Support**: Containerized deployment
- **Comprehensive Testing**: Unit and integration tests

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (optional, for containerized deployment)
- Tesseract OCR (for PDF processing)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yossufyasser1/Depi-grad-project--main.git
cd Depi-grad-project--main
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download required models**
```bash
# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download datasets (optional, for training)
python src/preprocessing/download_datasets.py
```

4. **Configure environment**
```bash
# Copy example environment file
cp .env .env.local

# Edit .env.local with your settings
```

5. **Run the API**
```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

6. **Access the application**
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# Stop
docker-compose down
```

### Using Docker Directly
```bash
# Build image
docker build -t study-assistant:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/chroma_db:/app/chroma_db \
  study-assistant:latest
```

## 📁 Project Structure

```
Depi-grad-project--main/
├── api/                        # FastAPI application
│   ├── main.py                # API endpoints and app configuration
│   └── schemas.py             # Pydantic models for validation
├── src/                       # Core application code
│   ├── preprocessing/         # Data preprocessing modules
│   │   ├── pdf_processor.py  # PDF text extraction with OCR
│   │   ├── text_preprocessor.py # NLP preprocessing
│   │   └── dataset_loader.py # Dataset loading utilities
│   ├── training/              # Model training modules
│   │   ├── textrank_summarizer.py    # Extractive summarization
│   │   ├── bart_summarizer.py        # Abstractive summarization
│   │   ├── bert_ner.py               # Named entity recognition
│   │   ├── keyword_extractor.py      # Keyword extraction
│   │   ├── t5_lora_trainer.py        # T5 fine-tuning with LoRA
│   │   └── distilbert_qa.py          # Question answering
│   ├── inference/             # Inference and RAG
│   │   ├── chromadb_manager.py       # Vector database management
│   │   ├── rag_retriever.py          # RAG document retrieval
│   │   └── llm_reader.py             # LLM answer generation
│   ├── config.py              # Configuration management
│   ├── evaluation_metrics.py # Evaluation utilities
│   ├── experiment_logger.py   # MLflow experiment tracking
│   └── performance_profiler.py # Performance benchmarking
├── tests/                     # Test suite
│   ├── test_unit.py          # Unit tests
│   ├── test_integration.py   # Integration tests
│   └── conftest.py           # PyTest configuration
├── data/                      # Data directory
│   ├── raw/                  # Raw uploaded documents
│   └── processed/            # Processed data
├── models/                    # Model storage
├── chroma_db/                 # Vector database storage
├── logs/                      # Application logs
├── mlruns/                    # MLflow experiment tracking
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Docker Compose configuration
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
├── .gitignore                # Git ignore rules
├── pytest.ini                # PyTest configuration
├── DEPLOYMENT_CHECKLIST.md   # Deployment guide
└── README.md                 # This file
```

## 🔧 Configuration

Edit the `.env` file to customize:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Model Paths
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=gpt2
SUMMARIZER_MODEL=facebook/bart-large-cnn

# Vector Database
TOP_K_RETRIEVAL=5
SIMILARITY_THRESHOLD=0.5

# Training
BATCH_SIZE=8
LEARNING_RATE=2e-5
NUM_EPOCHS=3
```

## 📚 API Endpoints

### Health Check
```bash
GET /health
```

### Upload Document
```bash
POST /upload
Content-Type: multipart/form-data

# Form data: file (PDF)
```

### Chat with Documents
```bash
POST /chat?query=What is machine learning?&top_k=5
```

### Summarize Text
```bash
POST /summarize?text=Your text here&summary_type=extractive
# summary_type: extractive or abstractive
```

### Extract Keywords
```bash
POST /extract-keywords?text=Your text here&top_n=10
```

### List Documents
```bash
GET /documents
```

### Delete Document
```bash
DELETE /documents/{document_id}
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Suites
```bash
# Unit tests only
pytest tests/test_unit.py -v

# Integration tests only
pytest tests/test_integration.py -v

# With coverage
pytest --cov=src --cov-report=html tests/
```

### Performance Benchmarking
```bash
# Run full benchmark suite
python src/performance_profiler.py

# Or import and run specific benchmarks
python -c "from src.performance_profiler import run_full_benchmark; run_full_benchmark()"
```

## 🎓 Model Training

### Fine-tune T5 with LoRA
```python
from src.training.t5_lora_trainer import T5LoRATrainer

trainer = T5LoRATrainer()
trainer.setup_model_with_lora()
trainer.fine_tune(train_dataset, eval_dataset)
```

### Train BERT NER
```python
from src.training.bert_ner import BertNERModel

model = BertNERModel()
model.fine_tune(train_dataset, eval_dataset)
```

### Track Experiments with MLflow
```python
from src.experiment_logger import ExperimentLogger

logger = ExperimentLogger()
logger.start_run("experiment_name")
logger.log_params({"learning_rate": 2e-5})
logger.log_metrics({"accuracy": 0.95})
logger.end_run()
```

## 📊 Evaluation

The project includes comprehensive evaluation metrics:

- **Summarization**: ROUGE, BLEU
- **NER**: F1, Precision, Recall (seqeval)
- **QA**: Exact Match, F1
- **Retrieval**: Precision@K, Recall@K, MRR, MAP

```python
from src.evaluation_metrics import EvaluationMetrics

metrics = EvaluationMetrics()
rouge_scores = metrics.compute_rouge(reference, hypothesis)
f1_score = metrics.compute_f1_ner(true_labels, pred_labels)
```

## 🛠️ Development

### Code Formatting
```bash
# Format code
black src/ api/ tests/

# Check formatting
black --check src/ api/ tests/
```

### Linting
```bash
flake8 src/ api/ tests/
```

### Pre-commit Hooks (Optional)
```bash
pip install pre-commit
pre-commit install
```

## 🔍 Troubleshooting

### Common Issues

**1. Import errors**
```bash
# Ensure you're in the project root and Python path is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**2. Tesseract not found**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows: Download installer from GitHub
```

**3. CUDA/GPU issues**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU-only if needed (edit .env)
USE_CUDA=False
```

**4. Memory issues**
```bash
# Reduce batch size in .env
BATCH_SIZE=4

# Or enable model quantization
```

## 📈 Performance

Target metrics on standard hardware:
- Document retrieval: < 500ms
- Extractive summarization: < 1s
- Abstractive summarization: < 5s
- Question answering: < 2s
- Keyword extraction: < 500ms

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is part of the DEPI graduation project.

## 👥 Authors

- Yossuf Yasser (@yossufyasser1)

## 🙏 Acknowledgments

- NotebookLM by Google for inspiration
- HuggingFace for transformer models
- FastAPI for the excellent web framework
- ChromaDB for vector storage

## 📞 Support

For issues, questions, or contributions:
- GitHub Issues: [Create an issue](https://github.com/yossufyasser1/Depi-grad-project--main/issues)
- Documentation: See `/docs` endpoint when running the API

## 🗺️ Roadmap

- [ ] Add support for more document formats (Word, Excel)
- [ ] Implement user authentication
- [ ] Add multi-language support
- [ ] Create web UI frontend
- [ ] Add batch processing capabilities
- [ ] Implement caching for faster responses
- [ ] Add support for larger LLMs (Llama, Mistral)
- [ ] Deploy to cloud platforms (AWS, GCP, Azure)

---

**Made with ❤️ for the DEPI graduation project**
