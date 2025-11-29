"""
Summarizer - Dedicated summarization module using pretrained models
Uses BART-large-CNN for high-quality text summarization 
"""
import os


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
from src.config import Config


class Summarizer:
    """Handles text summarization using pretrained models"""
    
    def __init__(self):
        """
        Initialize the summarization model and tokenizer
        Loads BART-large-CNN model (best quality)
        """
        print("📝 Initializing Summarizer...")
        
        # Use BART-large-CNN - state-of-the-art summarization
        model_name = Config.SUMMARIZATION_MODEL
        print(f"📦 Loading summarization model: {model_name}")
        print("⚠️ This is a large model (~1.6GB), first download may take time...")
        
        # Load tokenizer
        print("🔤 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load the model on CPU
        print("🧠 Loading model on CPU...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cpu"
        self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        print("✅ Summarizer initialized successfully!\n")
    
    def summarize_text(self, text, max_length=1000, min_length=350):
        """
        Summarize a given text using BART-large-CNN
        
        Args:
            text (str): Text to summarize
            max_length (int): Maximum summary length in tokens
            min_length (int): Minimum summary length in tokens
        
        Returns:
            str: Summarized text
        """
        print(f"📄 Summarizing text ({len(text)} characters)...")

        def clean_input(t: str) -> str:
            # Remove TOC dot leaders like "Chapter .... 12"
            t = re.sub(r"\s?\.{3,}\s?", " ", t)
            # Remove repeated page headers/footers patterns (section numbers at line start)
            t = re.sub(r"^(\d+\.\d+(\.\d+)?)\s+.+$", "", t, flags=re.MULTILINE)
            # De-hyphenate across line breaks
            t = re.sub(r"([\-–])\s*\n", r"\1", t)
            # Normalize whitespace/newlines
            t = re.sub(r"[ \t]+", " ", t)
            t = re.sub(r"\n{2,}", "\n", t)
            t = re.sub(r"\s{2,}", " ", t).strip()
            return t

        def clean_summary(s: str) -> str:
            s = re.sub(r'\.{2,}', '.', s)
            s = re.sub(r'\s{2,}', ' ', s).strip()
            return s
        
        # Clean noisy PDF artifacts before summarizing
        text = clean_input(text)

        # Hierarchical summarization for very long inputs
        max_input_length = 12000  # allow more input before chunking
        chunk_char_len = 4000
        overlap = 400
        
        def summarize_chunk(chunk: str) -> str:
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    length_penalty=1.1,
                    repetition_penalty=1.2,
                    num_beams=5,
                    early_stopping=True
                )
            return self.tokenizer.decode(ids[0], skip_special_tokens=True)
        
        try:
            print("🎯 Generating summary with BART-large-CNN (direct generate)...")
            # If very long, chunk then synthesize
            if len(text) > max_input_length:
                print("🔪 Chunking long input for hierarchical summarization...")
                chunks = []
                i = 0
                while i < len(text):
                    end = min(i + chunk_char_len, len(text))
                    chunks.append(text[i:end])
                    if end == len(text):
                        break
                    i = end - overlap
                partial_summaries = [clean_summary(summarize_chunk(c)) for c in chunks]
                combined = "\n\n".join(partial_summaries)
                print(f"🧩 Synthesizing final summary from {len(partial_summaries)} parts...")
                final = clean_summary(summarize_chunk(combined))
                print(f"✅ Summary generated ({len(final)} characters)\n")
                return final
            else:
                # Single-pass summary
                summary = clean_summary(summarize_chunk(text))
                print(f"✅ Summary generated ({len(summary)} characters)\n")
                return summary

        except Exception as e:
            print(f"⚠️ Summarization failed: {str(e)}\n")
            # Fallback: return truncated text
            return text[:max_length * 5] + "..."
