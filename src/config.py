from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Data paths
    RAW_DATA_PATH: Path = Path('./data/raw')
    PROCESSED_DATA_PATH: Path = Path('./data/processed')
    MODEL_SAVE_PATH: Path = Path('./models')
    CHECKPOINT_PATH: Path = Path('./models/checkpoints')
    
    # Dataset configurations
    SQUAD_DATASET: str = 'squad_v2'
    CONLL_DATASET: str = 'conll2003'
    TRAIN_RATIO: float = 0.8
    VAL_RATIO: float = 0.1
    TEST_RATIO: float = 0.1
    
    # Model configurations
    EMBEDDING_MODEL_NAME: str = 'all-MiniLM-L6-v2'
    LLM_MODEL_NAME: str = 'meta-llama/Llama-2-7b'
    SUMMARIZER_MODEL: str = 'facebook/bart-large-cnn'
    NER_MODEL_NAME: str = 'distilbert-base-uncased'
    QA_MODEL_NAME: str = 'distilbert-base-uncased-distilled-squad'
    
    # Training hyperparameters
    BATCH_SIZE: int = 8
    LEARNING_RATE: float = 2e-5
    NUM_EPOCHS: int = 3
    MAX_LENGTH: int = 512
    WARMUP_STEPS: int = 500
    
    # Vector Database
    VECTOR_DB_TYPE: str = 'chromadb'
    PERSIST_DIRECTORY: str = './chroma_db'
    
    # Embedding
    EMBEDDING_DIMENSION: int = 384
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    SIMILARITY_THRESHOLD: float = 0.5
    TOP_K_RETRIEVAL: int = 5
    
    # API Configuration
    API_HOST: str = '0.0.0.0'
    API_PORT: int = 8000
    API_WORKERS: int = 4
    LOG_LEVEL: str = 'INFO'
    
    # MLflow
    MLFLOW_TRACKING_URI: str = './mlruns'
    EXPERIMENT_NAME: str = 'ai_study_assistant'
    
    def __post_init__(self):
        for path in [self.RAW_DATA_PATH, self.PROCESSED_DATA_PATH, self.MODEL_SAVE_PATH]:
            path.mkdir(parents=True, exist_ok=True)

# Usage: from src.config import Config
# config = Config()
