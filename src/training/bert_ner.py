from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from seqeval.metrics import classification_report
import torch
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BertNERModel:
    def __init__(self, model_name: str = 'distilbert-base-uncased', num_labels: int = 9):
        """Initialize BERT for NER."""
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.info(f'Model loaded on {self.device}')
    
    def tokenize_and_align_labels(self, examples, label2id):
        """Align subword tokens with BIO labels."""
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=512,
            padding='max_length'
        )
        
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[label[word_idx]])
                else:
                    label_ids.append(label2id[label[word_idx]])
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def fine_tune(self, train_dataset, val_dataset, epochs: int = 3, batch_size: int = 8):
        """Fine-tune model on NER task."""
        training_args = TrainingArguments(
            output_dir='./models/bert_ner',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=2e-5,
            warmup_steps=500,
            weight_decay=0.01,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        logger.info('Fine-tuning completed')
    
    def predict(self, text: str) -> Dict:
        """Predict NER tags for text."""
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
        return {'text': text, 'predictions': predictions.tolist()}

# Usage:
# model = BertNERModel()
# model.fine_tune(train_dataset, val_dataset)
