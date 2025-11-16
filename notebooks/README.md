# 📚 Notebooks Guide

## Overview
This folder contains 6 specialized Jupyter notebooks for different aspects of the AI Study Assistant project.

## Notebooks

### 01_data_preprocessing.ipynb
**Purpose:** Data preparation and cleaning
- PDF text extraction with OCR
- Text preprocessing (tokenization, lemmatization)
- Dataset loading (SQuAD v2.0, CoNLL2003)
- Data quality validation

### 02_model_training.ipynb  
**Purpose:** Model training and fine-tuning
- Baseline models (TextRank, RAKE)
- BART abstractive summarization
- T5 + LoRA fine-tuning
- DistilBERT question answering
- BERT named entity recognition
- MLflow experiment tracking

### 03_vector_database.ipynb
**Purpose:** Vector database management
- ChromaDB initialization
- Document embedding generation
- Semantic search operations
- CRUD operations
- Metadata filtering

### 04_rag_pipeline.ipynb
**Purpose:** Retrieval-Augmented Generation
- RAG retriever setup
- Multi-query retrieval
- LLM reader for answer generation
- Streaming responses
- End-to-end RAG workflow

### 05_evaluation.ipynb
**Purpose:** Model evaluation metrics
- ROUGE scores for summarization
- BLEU scores
- F1 scores for NER
- Exact Match for QA
- Retrieval metrics (P@K, R@K, MRR, MAP)

### 06_performance_optimization.ipynb
**Purpose:** Performance profiling and optimization
- Execution time profiling
- Memory usage tracking
- Model benchmarking
- System resource monitoring
- Optimization strategies

## Usage

1. **Sequential Learning**: Work through notebooks in order (01 → 06)
2. **Targeted Practice**: Jump to specific notebooks for focused tasks
3. **Experimentation**: Modify code and parameters to explore

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Download models
python -m spacy download en_core_web_sm

# Download datasets (optional)
python src/preprocessing/download_datasets.py
```

## Tips

- Run cells sequentially for best results
- Check cell outputs for errors
- Save your work frequently
- Use the main `tutorial.ipynb` for a comprehensive overview

## Next Steps

After completing these notebooks:
1. Test the FastAPI endpoints (see GETTING_STARTED.py)
2. Deploy using Docker (see DEPLOYMENT_CHECKLIST.md)
3. Build your own applications using the trained models

Happy coding! 🚀
