from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import torch
from pathlib import Path
from typing import Dict, Optional
import logging

# Import project config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BARTSummarizer:
    """BART model trainer for abstractive summarization."""
    
    def __init__(self, model_name: str = 'facebook/bart-large-cnn', config: Optional[Config] = None):
        """Initialize BART summarization model."""
        self.model_name = model_name
        self.config = config or Config()
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')
    
    def load_model(self):
        """Load BART model and tokenizer."""
        logger.info(f'Loading {self.model_name}...')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f'Total parameters: {total_params:,}')
        logger.info(f'Trainable parameters: {trainable_params:,}')
        
        return self.model, self.tokenizer
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """Preprocess examples for BART summarization."""
        # BART doesn't need task prefix like T5, but we can add it for consistency
        inputs = examples['article']
        
        # Tokenize inputs with BART's max length (1024)
        model_inputs = self.tokenizer(
            inputs,
            max_length=1024,
            truncation=True,
            padding='max_length'
        )
        
        # Tokenize targets (summaries)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples['highlights'],
                max_length=256,
                truncation=True,
                padding='max_length'
            )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    def fine_tune(self, dataset_name: str = 'cnn_dailymail', dataset_version: str = '3.0.0'):
        """Fine-tune BART for abstractive summarization."""
        # Load model if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Load dataset
        logger.info(f'Loading {dataset_name} dataset (version {dataset_version})...')
        dataset = load_dataset(dataset_name, dataset_version)
        
        # Use subset for faster training/testing (remove for full training)
        train_size = min(10000, len(dataset['train']))
        val_size = min(1000, len(dataset['validation']))
        
        dataset['train'] = dataset['train'].select(range(train_size))
        dataset['validation'] = dataset['validation'].select(range(val_size))
        
        logger.info(f'Using {train_size} training samples and {val_size} validation samples')
        
        # Preprocess dataset
        logger.info('Preprocessing dataset...')
        processed_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=['article', 'highlights', 'id'],
            desc='Tokenizing'
        )
        
        # Create output directory
        output_dir = self.config.MODEL_SAVE_PATH / 'bart_summarizer'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training arguments optimized for BART
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=2,  # BART is pre-trained, fewer epochs needed
            per_device_train_batch_size=4,  # Smaller batch for large model
            per_device_eval_batch_size=4,
            learning_rate=3e-5,  # Lower learning rate for fine-tuning
            warmup_steps=1000,
            weight_decay=0.01,
            gradient_accumulation_steps=4,  # Effective batch size = 16
            evaluation_strategy='steps',
            eval_steps=500,
            save_strategy='steps',
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            logging_steps=100,
            logging_dir=str(self.config.MODEL_SAVE_PATH / 'logs' / 'bart'),
            report_to='none',
            fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
            save_total_limit=3,  # Keep only last 3 checkpoints
            prediction_loss_only=True,
        )
        
        # Data collator for dynamic padding and label preparation
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_dataset['train'],
            eval_dataset=processed_dataset['validation'],
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Start training
        logger.info('Starting BART fine-tuning...')
        trainer.train()
        
        # Save final model
        save_path = output_dir / 'final_model'
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        
        logger.info(f'BART fine-tuning completed! Model saved to {save_path}')
        
        return trainer
    
    def generate_summary(
        self,
        text: str,
        max_length: int = 256,
        min_length: int = 56,
        num_beams: int = 4,
        length_penalty: float = 2.0
    ) -> str:
        """Generate summary using fine-tuned BART."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.model.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=True
            )
        
        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def load_pretrained(self, model_path: str):
        """Load a previously fine-tuned model."""
        logger.info(f'Loading fine-tuned model from {model_path}...')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        logger.info('Model loaded successfully!')

# Usage:
# summarizer = BARTSummarizer()
# summarizer.fine_tune()
# summary = summarizer.generate_summary("Your long article text here...")
#
# Or load a pre-trained model:
# summarizer.load_pretrained('./models/bart_summarizer/final_model')
# summary = summarizer.generate_summary("Your text...")
