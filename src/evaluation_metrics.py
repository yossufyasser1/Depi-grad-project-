from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
import logging

# Import project config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """Comprehensive evaluation metrics for NLP tasks."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize evaluation metrics."""
        self.config = config or Config()
        logger.info('Evaluation metrics initialized')
    
    @staticmethod
    def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE scores for summarization evaluation.
        
        Args:
            predictions: List of generated summaries
            references: List of reference summaries
            
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        logger.info(f'Computing ROUGE scores for {len(predictions)} predictions...')
        
        try:
            from evaluate import load
            rouge = load('rouge')
            results = rouge.compute(predictions=predictions, references=references)
            
            logger.info(f'ROUGE-1: {results.get("rouge1", 0):.4f}')
            logger.info(f'ROUGE-2: {results.get("rouge2", 0):.4f}')
            logger.info(f'ROUGE-L: {results.get("rougeL", 0):.4f}')
            
            return results
        except ImportError:
            logger.warning('evaluate library not found, using basic ROUGE approximation')
            return EvaluationMetrics._compute_rouge_basic(predictions, references)
    
    @staticmethod
    def _compute_rouge_basic(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Basic ROUGE-1 approximation without evaluate library."""
        from collections import Counter
        
        rouge1_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            pred_counter = Counter(pred_tokens)
            ref_counter = Counter(ref_tokens)
            
            overlap = sum((pred_counter & ref_counter).values())
            
            if len(ref_tokens) > 0:
                rouge1 = overlap / len(ref_tokens)
                rouge1_scores.append(rouge1)
        
        return {
            'rouge1': float(np.mean(rouge1_scores)) if rouge1_scores else 0.0,
            'rouge2': 0.0,  # Placeholder
            'rougeL': 0.0   # Placeholder
        }
    
    @staticmethod
    def compute_bleu(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        Compute BLEU score for text generation.
        
        Args:
            predictions: List of generated texts
            references: List of reference texts (can have multiple references per prediction)
            
        Returns:
            Dictionary with BLEU score
        """
        logger.info(f'Computing BLEU score for {len(predictions)} predictions...')
        
        try:
            from evaluate import load
            bleu = load('bleu')
            results = bleu.compute(predictions=predictions, references=references)
            
            logger.info(f'BLEU: {results.get("bleu", 0):.4f}')
            return results
        except ImportError:
            logger.warning('evaluate library not found, skipping BLEU computation')
            return {'bleu': 0.0}
    
    @staticmethod
    def compute_f1_ner(predictions: List[List[int]], labels: List[List[int]], label_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute F1 score for Named Entity Recognition.
        
        Args:
            predictions: List of predicted label sequences
            labels: List of true label sequences
            label_names: Optional list of label names for detailed metrics
            
        Returns:
            Dictionary with F1, precision, recall scores
        """
        logger.info(f'Computing NER F1 score for {len(predictions)} sequences...')
        
        try:
            from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
            
            # Convert integer labels to string labels if needed
            if label_names:
                pred_labels = [[label_names[p] for p in pred] for pred in predictions]
                true_labels = [[label_names[l] for l in lbl] for lbl in labels]
            else:
                pred_labels = [[str(p) for p in pred] for pred in predictions]
                true_labels = [[str(l) for l in lbl] for lbl in labels]
            
            f1 = f1_score(true_labels, pred_labels)
            precision = precision_score(true_labels, pred_labels)
            recall = recall_score(true_labels, pred_labels)
            
            logger.info(f'F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
            
            # Get detailed report
            report = classification_report(true_labels, pred_labels, output_dict=True)
            
            return {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'detailed_report': report
            }
        except ImportError:
            logger.warning('seqeval library not found, using basic F1 computation')
            return EvaluationMetrics._compute_f1_basic(predictions, labels)
    
    @staticmethod
    def _compute_f1_basic(predictions: List[List[int]], labels: List[List[int]]) -> Dict[str, float]:
        """Basic F1 computation without seqeval."""
        correct = 0
        total_pred = 0
        total_true = 0
        
        for pred, true in zip(predictions, labels):
            for p, t in zip(pred, true):
                if p != -100:  # Ignore padding
                    total_pred += 1
                    if t != -100:
                        total_true += 1
                        if p == t:
                            correct += 1
        
        precision = correct / total_pred if total_pred > 0 else 0.0
        recall = correct / total_true if total_true > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    @staticmethod
    def compute_exact_match(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute exact match and F1 for Question Answering.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            Dictionary with exact match and F1 scores
        """
        logger.info(f'Computing exact match for {len(predictions)} predictions...')
        
        exact_matches = []
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            # Normalize texts
            pred_normalized = pred.strip().lower()
            ref_normalized = ref.strip().lower()
            
            # Exact match
            em = 1.0 if pred_normalized == ref_normalized else 0.0
            exact_matches.append(em)
            
            # Token-level F1
            pred_tokens = set(pred_normalized.split())
            ref_tokens = set(ref_normalized.split())
            
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                f1 = 0.0
            else:
                common = pred_tokens & ref_tokens
                precision = len(common) / len(pred_tokens)
                recall = len(common) / len(ref_tokens)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            f1_scores.append(f1)
        
        em_score = float(np.mean(exact_matches))
        f1_score_val = float(np.mean(f1_scores))
        
        logger.info(f'Exact Match: {em_score:.4f}, F1: {f1_score_val:.4f}')
        
        return {
            'exact_match': em_score,
            'f1': f1_score_val
        }
    
    @staticmethod
    def compute_retrieval_metrics(
        retrieved_docs: List[List[str]],
        relevant_docs: List[List[str]],
        k_values: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics (Precision, Recall, F1, MRR, MAP).
        
        Args:
            retrieved_docs: List of retrieved document lists for each query
            relevant_docs: List of relevant document lists for each query
            k_values: List of k values for Precision@k and Recall@k
            
        Returns:
            Dictionary with retrieval metrics
        """
        logger.info(f'Computing retrieval metrics for {len(retrieved_docs)} queries...')
        
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        precisions = []
        recalls = []
        mrr_scores = []
        ap_scores = []
        
        precision_at_k = {k: [] for k in k_values}
        recall_at_k = {k: [] for k in k_values}
        
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            # Convert to sets for comparison (use first 100 chars as ID)
            retrieved_set = set([d[:100] for d in retrieved])
            relevant_set = set([d[:100] for d in relevant])
            
            # Overall Precision and Recall
            if len(retrieved_set) > 0:
                precision = len(retrieved_set & relevant_set) / len(retrieved_set)
                precisions.append(precision)
            
            if len(relevant_set) > 0:
                recall = len(retrieved_set & relevant_set) / len(relevant_set)
                recalls.append(recall)
            
            # Mean Reciprocal Rank (MRR)
            for i, doc in enumerate(retrieved):
                if doc[:100] in relevant_set:
                    mrr_scores.append(1.0 / (i + 1))
                    break
            else:
                mrr_scores.append(0.0)
            
            # Average Precision (AP)
            ap = 0.0
            relevant_count = 0
            for i, doc in enumerate(retrieved):
                if doc[:100] in relevant_set:
                    relevant_count += 1
                    ap += relevant_count / (i + 1)
            
            if len(relevant_set) > 0:
                ap /= len(relevant_set)
            ap_scores.append(ap)
            
            # Precision@k and Recall@k
            for k in k_values:
                retrieved_k = set([d[:100] for d in retrieved[:k]])
                
                if len(retrieved_k) > 0:
                    precision_k = len(retrieved_k & relevant_set) / len(retrieved_k)
                    precision_at_k[k].append(precision_k)
                
                if len(relevant_set) > 0:
                    recall_k = len(retrieved_k & relevant_set) / len(relevant_set)
                    recall_at_k[k].append(recall_k)
        
        # Compute averages
        avg_precision = float(np.mean(precisions)) if precisions else 0.0
        avg_recall = float(np.mean(recalls)) if recalls else 0.0
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        mrr = float(np.mean(mrr_scores)) if mrr_scores else 0.0
        map_score = float(np.mean(ap_scores)) if ap_scores else 0.0
        
        results = {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': f1,
            'mrr': mrr,
            'map': map_score
        }
        
        # Add Precision@k and Recall@k
        for k in k_values:
            results[f'precision@{k}'] = float(np.mean(precision_at_k[k])) if precision_at_k[k] else 0.0
            results[f'recall@{k}'] = float(np.mean(recall_at_k[k])) if recall_at_k[k] else 0.0
        
        logger.info(f'Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {f1:.4f}')
        logger.info(f'MRR: {mrr:.4f}, MAP: {map_score:.4f}')
        
        return results
    
    @staticmethod
    def compute_perplexity(loss: float) -> float:
        """
        Compute perplexity from loss.
        
        Args:
            loss: Cross-entropy loss
            
        Returns:
            Perplexity score
        """
        perplexity = np.exp(loss)
        logger.info(f'Perplexity: {perplexity:.4f}')
        return perplexity
    
    @staticmethod
    def compute_classification_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
        """
        Compute classification metrics (accuracy, precision, recall, F1).
        
        Args:
            predictions: List of predicted labels
            labels: List of true labels
            
        Returns:
            Dictionary with classification metrics
        """
        logger.info(f'Computing classification metrics for {len(predictions)} predictions...')
        
        # Accuracy
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        accuracy = correct / len(predictions) if len(predictions) > 0 else 0.0
        
        # For binary or multi-class, compute precision, recall, F1
        tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        logger.info(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

# Usage:
# metrics = EvaluationMetrics()
#
# # Summarization
# rouge = metrics.compute_rouge(predictions, references)
#
# # NER
# f1_results = metrics.compute_f1_ner(predictions, labels, label_names)
#
# # QA
# qa_results = metrics.compute_exact_match(predictions, references)
#
# # Retrieval
# retrieval_results = metrics.compute_retrieval_metrics(retrieved_docs, relevant_docs)
