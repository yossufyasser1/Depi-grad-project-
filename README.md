# AI Study Assistant (Backend)

Comprehensive documentation for the FastAPI backend implementing local-first Retrieval-Augmented Generation (RAG), summarization, and Q&A generation.

## Overview
- Purpose: Answer study questions from your documents with RAG, generate summaries, and create Q&A pairs.
- Local-first: Retrieval uses FAISS + sentence-transformers locally. Generation uses Ollama exclusively.
- Stack: FastAPI, LangChain, FAISS, sentence-transformers, Transformers (PyTorch), Ollama.

## Architecture
- `main.py`: FastAPI app and endpoints.
- `src/config.py`: Central configuration for paths, models, providers.
- `src/processing/pdf_processor.py`: Extracts text/chunks from PDFs.
- `src/processing/improved_rag_retriever.py`: Builds/queries FAISS with local embeddings.
- `src/inference/improved_llm_reader.py`: Ollama-based LLM for answers.
- `src/inference/summarizer.py`: BART-large-CNN summarization with hierarchical synthesis.
- `src/inference/qa_generator.py`: FLAN‑T5 Q&A generation from plain text.

## Data Flow & Workflows
- Upload & Index
   - Client posts a PDF to `/upload`.
   - PDF is parsed → text → chunked.
   - Embeddings computed via `sentence-transformers` and stored into FAISS (`vector_db/`).
- Chat (RAG)
   - Client posts a question to `/chat`.
   - If FAISS has documents: retrieve top-k chunks → format a `context` string.
   - `ImprovedLLMReader` builds a prompt with `question + context + short history` and generates the answer.
- Chat (No RAG)
   - If FAISS empty: `generate_simple_answer` produces a general answer via provider.
- Summarization
   - Client posts text to `/summarize`.
   - Input cleaning (remove TOC dots, headers/footers, hyphenations) → chunk summaries → synthesized final summary.
- Q&A Generation
   - Client posts text to `/generate-qa`.
   - FLAN‑T5 creates a set of Q&A pairs from the text.

## Endpoints
- `POST /upload`: Upload a single PDF; indexes chunks.
- `POST /upload-with-qa?num_questions=10`: Upload PDF, generate Q&A, then index.
- `POST /chat`: Ask a question; uses RAG context when available.
- `POST /summarize`: Summarize text (`max_length`, `min_length` supported).
- `POST /generate-qa`: Generate Q&A pairs from text.
- `GET /health`: Server health + document count.
- `GET /rag-stats`: RAG stats + conversation metadata.
- `DELETE /documents`: Clear FAISS cache and reset.

## Configuration (`src/config.py`)
- Paths
   - `UPLOAD_DIR`: `data/uploads`
   - `FAISS_DB_DIR`: `vector_db`
- RAG
   - `EMBEDDING_MODEL`: `sentence-transformers/all-mpnet-base-v2`
   - `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `RAG_TOP_K`
   - `RAG_TEMPERATURE`, `RAG_MAX_OUTPUT_TOKENS`
- Generation Providers
   - Ollama (local): `OLLAMA_ENABLED`, `OLLAMA_MODEL` (e.g., `llama3.2:latest`), `OLLAMA_BASE_URL`
- Tasks
   - Summarization: `SUMMARIZATION_MODEL = "facebook/bart-large-cnn"`
   - Q&A: `QA_MODEL = "google/flan-t5-base"`

## Models & Rationale
- Embeddings: `all-mpnet-base-v2` — strong semantic retrieval; CPU-friendly.
- Vector store: FAISS — fast similarity search; persisted locally.
- LLM (Chat):
   - Ollama (`llama3.2:latest` or other models) — private, local inference.
- Summarization: BART-large-CNN — high quality abstractive summaries.
- Q&A: FLAN‑T5‑base — effective instruction-tuned generation for pairs.

## Summarization Strategy
- Cleaning: remove TOC dot leaders, repeated section headers, page artifacts, and de-hyphenate lines.
- Parameters: defaults `max_length=1000`, `min_length=350`, `num_beams=5`, `repetition_penalty=1.2`.
- Hierarchical: chunk long inputs (~4000 chars, overlap 400), summarize per-chunk, then synthesize a final summary.

## Provider Selection (LLM)
- `ImprovedLLMReader` uses Ollama exclusively:
   - Uses `langchain_community.llms.Ollama` and `invoke(prompt)`.
   - Requires Ollama running locally at `OLLAMA_BASE_URL`.
- Conversation: recent messages are included for coherence; name detection heuristics applied.

## RAG Details
- Document loading: `PyPDFLoader`, `Docx2txtLoader`, `TextLoader`, `UnstructuredMarkdownLoader`.
- Chunking: `RecursiveCharacterTextSplitter` using `RAG_CHUNK_SIZE` and `RAG_CHUNK_OVERLAP`.
- Indexing: `FAISS.from_documents(chunks, embeddings)` and `save_local(cache_dir)`.
- Retrieval: `similarity_search` (or `_with_score`) → format context with `---DOCUMENT SEPARATOR---`.

## Setup
- Python 3.10–3.12 recommended.
- Install dependencies:
   - `pip install langchain langchain-community sentence-transformers faiss-cpu pypdf docx2txt unstructured`
   - `pip install transformers`
   - CPU PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
   - Optional Gemini: `pip install google-generativeai`
- Ollama (Windows):
   - Install via official installer: https://ollama.com/download/windows
   - Ensure API reachable: `curl http://localhost:11434/api/version`
   - Check models: `curl http://localhost:11434/api/tags`
- Configure `src/config.py`:
   - Enable Ollama: `OLLAMA_ENABLED=True`, set `OLLAMA_MODEL` (e.g., `llama3.2:latest`).

## Run
- Development:
   - `python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000`
   - Or `python main.py` (no auto-reload unless configured).
- Open docs: `http://localhost:8000/docs`

## Usage Examples (CMD-friendly)
- Upload a PDF:
   - `curl -X POST "http://localhost:8000/upload" -H "Content-Type: application/pdf" --data-binary @"Y:\\projects\\Depi-grad-project- - Copy\\data\\uploads\\sample.pdf"`
- Chat with RAG:
   - `curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"question\":\"Explain chapter 2\",\"top_k\":5}"`
- Summarize text:
   - `curl -X POST http://localhost:8000/summarize -H "Content-Type: application/json" -d "{\"text\":\"<paste text>\",\"max_length\":1000,\"min_length\":350}"`
- Health/Stats:
   - `curl http://localhost:8000/health`
   - `curl http://localhost:8000/rag-stats`

## Operational Tips
- Prefer smaller Ollama models on CPU for speed (e.g., `llama3.2:3b`).
- Keep `top_k` modest (3–5); too much context hurts focus and speed.
- Cap context length in `ImprovedRAGRetriever.format_context` if latency grows.
- If FAISS/cpu wheel issues occur on specific Python versions, consider upgrading Python or using an alternative vector store temporarily.

## Troubleshooting
- Ollama CLI missing but API works:
   - Use full path (`"C:\\Program Files\\Ollama\\ollama.exe" serve`) or only HTTP API.
   - Add to PATH: `setx PATH "%PATH%;C:\\Program Files\\Ollama"` then reopen terminal.
- Gemini rate limits:
   - Ensure `OLLAMA_ENABLED=True` to avoid cloud dependence; keep retrieval local.
- Transformers import issues (TensorFlow/JAX):
   - We set `TRANSFORMERS_NO_TF/JAX/FLAX` env flags in `summarizer.py` and avoid pipelines path.

## Future Enhancements
- PDFProcessor cleanup for headers/footers and page numbering.
- Context length caps and reranking.
- Streaming responses for chat.
- Model registry in config for easier switching and versioning.

## License
- Proprietary project files. Do not distribute models or PDFs included in your environment.
