import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_all_datasets():
    """Download all required datasets."""
    from src.preprocessing.dataset_loader import DatasetLoader
    
    # Create data directory
    raw_data_dir = Path('./data/raw')
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info('Starting dataset downloads...')
    
    # Download SQuAD v2.0
    logger.info('Downloading SQuAD v2.0 (train)...')
    squad_loader = DatasetLoader('squad_v2', split='train')
    squad_data = squad_loader.load()
    squad_loader.save_processed_dataset(str(raw_data_dir / 'squad_v2'))
    logger.info('✓ SQuAD v2.0 downloaded')
    
    # Download SQuAD v2.0 validation
    logger.info('Downloading SQuAD v2.0 (validation)...')
    squad_val = DatasetLoader('squad_v2', split='validation')
    squad_val_data = squad_val.load()
    squad_val.save_processed_dataset(str(raw_data_dir / 'squad_v2_val'))
    logger.info('✓ SQuAD v2.0 validation downloaded')
    
    # Download CoNLL-2003
    logger.info('Downloading CoNLL-2003 (train)...')
    conll_loader = DatasetLoader('conll2003', split='train')
    conll_data = conll_loader.load()
    conll_loader.save_processed_dataset(str(raw_data_dir / 'conll2003'))
    logger.info('✓ CoNLL-2003 downloaded')
    
    logger.info('All datasets downloaded successfully!')

if __name__ == '__main__':
    download_all_datasets()
