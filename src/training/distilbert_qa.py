from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DefaultDataCollator
)
from datasets import load_dataset
import torch
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import logging

# Import project config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistilBERTQA:
    """DistilBERT model trainer for Question Answering."""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased', config: Optional[Config] = None):
        """Initialize DistilBERT for Question Answering."""
        self.model_name = model_name
        self.config = config or Config()
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')
    
    def load_model(self):
        """Load DistilBERT model and tokenizer for QA."""
        logger.info(f'Loading {self.model_name} for Question Answering...')
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f'Total parameters: {total_params:,}')
        logger.info(f'Trainable parameters: {trainable_params:,}')
        
        return self.model, self.tokenizer
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """Preprocess examples for Question Answering task."""
        # Strip whitespace from questions
        questions = [q.strip() for q in examples['question']]
        
        # Tokenize questions and contexts together
        inputs = self.tokenizer(
            questions,
            examples['context'],
            max_length=384,
            truncation='only_second',  # Only truncate context, not question
            return_offsets_mapping=True,
            padding='max_length',
            return_tensors=None
        )
        
        # Get offset mapping and answers
        offset_mapping = inputs.pop('offset_mapping')
        answers = examples['answers']
        start_positions = []
        end_positions = []
        
        # Process each example
        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            
            # Handle unanswerable questions (SQuAD v2.0)
            if len(answer['answer_start']) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue
            
            # Get answer position in original text
            start_char = answer['answer_start'][0]
            end_char = start_char + len(answer['text'][0])
            
            # Get sequence IDs to identify context tokens
            sequence_ids = inputs.sequence_ids(i)
            
            # Find start and end of context in token sequence
            idx = 0
            while idx < len(sequence_ids) and sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1
            
            # Check if answer is fully in context (not truncated)
            if offset[context_start][0] <= start_char and offset[context_end][1] >= end_char:
                # Find token positions for answer
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)
                
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)
            else:
                # Answer is truncated, mark as unanswerable
                start_positions.append(0)
                end_positions.append(0)
        
        inputs['start_positions'] = start_positions
        inputs['end_positions'] = end_positions
        
        return inputs
    
    def fine_tune(self, dataset_name: str = 'squad_v2'):
        """Fine-tune DistilBERT on SQuAD dataset."""
        # Load model if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Load dataset
        logger.info(f'Loading {dataset_name} dataset...')
        dataset = load_dataset(dataset_name)
        
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
            remove_columns=dataset['train'].column_names,
            desc='Tokenizing'
        )
        
        # Create output directory
        output_dir = self.config.MODEL_SAVE_PATH / 'distilbert_qa'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=16,  # Larger batch for QA task
            per_device_eval_batch_size=16,
            learning_rate=3e-5,
            warmup_steps=1000,
            weight_decay=0.01,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            logging_steps=100,
            logging_dir=str(self.config.MODEL_SAVE_PATH / 'logs' / 'distilbert_qa'),
            report_to='none',
            fp16=torch.cuda.is_available(),
            save_total_limit=2,
        )
        
        # Data collator
        data_collator = DefaultDataCollator()
        
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
        logger.info('Starting DistilBERT QA fine-tuning...')
        trainer.train()
        
        # Save final model
        save_path = output_dir / 'final_model'
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        
        logger.info(f'QA fine-tuning completed! Model saved to {save_path}')
        
        return trainer
    
    def answer_question(self, question: str, context: str, top_k: int = 1) -> List[Dict]:
        """Answer a question given a context."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.model.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            question,
            context,
            max_length=384,
            truncation='only_second',
            return_tensors='pt',
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get start and end logits
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
        
        # Get top k answers
        answers = []
        
        # Get token-level predictions
        start_idx = torch.argmax(start_logits).item()
        end_idx = torch.argmax(end_logits).item()
        
        # Ensure valid span
        if start_idx <= end_idx and end_idx < len(inputs['input_ids'][0]):
            answer_tokens = inputs['input_ids'][0][start_idx:end_idx + 1]
            answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            # Calculate confidence score
            confidence = (start_logits[start_idx] + end_logits[end_idx]).item() / 2
            
            answers.append({
                'answer': answer_text,
                'confidence': confidence,
                'start_idx': start_idx,
                'end_idx': end_idx
            })
        else:
            # No valid answer found
            answers.append({
                'answer': '',
                'confidence': 0.0,
                'start_idx': -1,
                'end_idx': -1
            })
        
        return answers
    
    def load_pretrained(self, model_path: str):
        """Load a previously fine-tuned model."""
        logger.info(f'Loading fine-tuned model from {model_path}...')
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        logger.info('Model loaded successfully!')

# Usage:
# qa_model = DistilBERTQA()
# qa_model.fine_tune()
# answer = qa_model.answer_question("What is AI?", "Artificial Intelligence is...")
#
# Or load a pre-trained model:
# qa_model.load_pretrained('./models/distilbert_qa/final_model')
# answer = qa_model.answer_question("Your question", "Your context")
