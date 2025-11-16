from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch
from pathlib import Path
from typing import Dict, Tuple
import logging

# Import project config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class T5LoRATrainer:
    """T5 model trainer with LoRA for efficient fine-tuning."""
    
    def __init__(self, model_name: str = 't5-base', config: Config = None):
        """Initialize T5 with LoRA configuration."""
        self.model_name = model_name
        self.config = config or Config()
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')
    
    def setup_model_with_lora(self) -> Tuple[torch.nn.Module, AutoTokenizer]:
        """Load T5 and apply LoRA for parameter-efficient fine-tuning."""
        logger.info(f'Loading {self.model_name}...')
        
        # Load base model and tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # LoRA Configuration for efficient training
        lora_config = LoraConfig(
            r=8,  # Low-rank dimension
            lora_alpha=32,  # Scaling factor
            target_modules=['q', 'v'],  # Target attention layers
            lora_dropout=0.05,
            bias='none',
            task_type='SEQ_2_SEQ_LM'
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Calculate trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        trainable_percent = 100 * trainable_params / all_params
        
        logger.info(f'Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}%)')
        logger.info(f'Total parameters: {all_params:,}')
        
        return self.model, self.tokenizer
    
    def preprocess_function(self, examples: Dict, source_prefix: str = 'summarize: ') -> Dict:
        """Preprocess examples for T5 summarization."""
        # Add task prefix for T5
        inputs = [source_prefix + doc for doc in examples['article']]
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config.MAX_LENGTH,
            truncation=True,
            padding='max_length'
        )
        
        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples['highlights'],
                max_length=128,
                truncation=True,
                padding='max_length'
            )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    def fine_tune(self, dataset_name: str = 'cnn_dailymail', dataset_version: str = '3.0.0'):
        """Fine-tune T5 with LoRA for summarization task."""
        # Setup model if not already done
        if self.model is None or self.tokenizer is None:
            self.setup_model_with_lora()
        
        # Load dataset
        logger.info(f'Loading {dataset_name} dataset (version {dataset_version})...')
        dataset = load_dataset(dataset_name, dataset_version)
        
        # Take a subset for faster training (remove this for full training)
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
        output_dir = self.config.MODEL_SAVE_PATH / 't5_lora_summarizer'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            learning_rate=self.config.LEARNING_RATE,
            warmup_steps=self.config.WARMUP_STEPS,
            weight_decay=0.01,
            evaluation_strategy='steps',
            eval_steps=500,
            save_strategy='steps',
            save_steps=500,
            load_best_model_at_end=True,
            logging_steps=100,
            logging_dir=str(self.config.MODEL_SAVE_PATH / 'logs'),
            report_to='none',  # Disable wandb/tensorboard if not configured
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        )
        
        # Data collator for dynamic padding
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
        logger.info('Starting fine-tuning...')
        trainer.train()
        
        # Save model
        save_path = output_dir / 'final_model'
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        
        logger.info(f'Fine-tuning completed! Model saved to {save_path}')
        
        return trainer
    
    def generate_summary(self, text: str, max_length: int = 128) -> str:
        """Generate summary for input text."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call setup_model_with_lora() first.")
        
        self.model.eval()
        self.model.to(self.device)
        
        # Prepare input
        input_text = f"summarize: {text}"
        inputs = self.tokenizer(
            input_text,
            max_length=self.config.MAX_LENGTH,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary

# Usage:
# trainer = T5LoRATrainer(model_name='t5-base')
# trainer.fine_tune()
# summary = trainer.generate_summary("Your long text here...")
