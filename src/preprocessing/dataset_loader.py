from datasets import load_dataset, DatasetDict
from typing import Dict, List, Tuple
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetLoader:
    def __init__(self, dataset_type: str, split: str = 'train'):
        """Initialize dataset loader."""
        self.dataset_type = dataset_type
        self.split = split
        self.dataset = None
    
    def load_squad_v2(self) -> Dict:
        """Load SQuAD v2.0 dataset."""
        logger.info(f'Loading SQuAD v2.0 ({self.split} split)...')
        dataset = load_dataset('squad_v2', split=self.split)
        logger.info(f'Loaded {len(dataset)} samples')
        
        # Convert to dict format
        processed = {
            'contexts': [],
            'questions': [],
            'answers': []
        }
        
        for item in dataset:
            processed['contexts'].append(item['context'])
            processed['questions'].append(item['question'])
            processed['answers'].append(item['answers'])
        
        return processed
    
    def load_conll2003(self) -> Dict:
        """Load CoNLL-2003 NER dataset."""
        logger.info(f'Loading CoNLL-2003 ({self.split} split)...')
        dataset = load_dataset('conll2003', split=self.split)
        logger.info(f'Loaded {len(dataset)} samples')
        
        # Convert to dict format
        processed = {
            'tokens': [],
            'ner_tags': [],
            'pos_tags': []
        }
        
        for item in dataset:
            processed['tokens'].append(item['tokens'])
            processed['ner_tags'].append(item['ner_tags'])
            processed['pos_tags'].append(item['pos_tags'])
        
        return processed
    
    def load_custom_dataset(self, file_path: str) -> Dict:
        """Load custom JSON dataset."""
        logger.info(f'Loading custom dataset from {file_path}...')
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f'Loaded {len(data)} samples')
        return data
    
    def preprocess_batch(self, batch: Dict, task_type: str) -> Dict:
        """Preprocess batch based on task type."""
        if task_type == 'qa':
            # Question answering preprocessing
            return {
                'contexts': batch.get('contexts', []),
                'questions': batch.get('questions', []),
                'answers': batch.get('answers', [])
            }
        elif task_type == 'ner':
            # NER preprocessing
            return {
                'tokens': batch.get('tokens', []),
                'ner_tags': batch.get('ner_tags', [])
            }
        else:
            return batch
    
    def get_data_stats(self) -> Dict:
        """Get dataset statistics."""
        if self.dataset is None:
            if self.dataset_type == 'squad_v2':
                self.dataset = self.load_squad_v2()
            elif self.dataset_type == 'conll2003':
                self.dataset = self.load_conll2003()
        
        stats = {
            'num_samples': len(self.dataset.get(list(self.dataset.keys())[0], [])),
            'keys': list(self.dataset.keys())
        }
        logger.info(f'Dataset stats: {stats}')
        return stats
    
    def save_processed_dataset(self, output_path: str):
        """Save processed dataset to disk."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'data.json', 'w') as f:
            json.dump(self.dataset, f, indent=2)
        
        logger.info(f'Saved dataset to {output_path}')
    
    def load(self) -> Dict:
        """Load dataset based on type."""
        if self.dataset_type == 'squad_v2':
            self.dataset = self.load_squad_v2()
        elif self.dataset_type == 'conll2003':
            self.dataset = self.load_conll2003()
        else:
            raise ValueError(f'Unknown dataset type: {self.dataset_type}')
        
        return self.dataset

# Usage:
# loader = DatasetLoader('squad_v2', split='train')
# data = loader.load()
