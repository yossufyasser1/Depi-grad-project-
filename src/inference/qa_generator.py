"""
Question Answer Generator - Automatically generates Q&A pairs from text
Uses FLAN-T5-base model for better question generation
"""
import os
# Disable TensorFlow to prevent NumPy compatibility issues
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TF'] = '0'

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
from src.config import Config


class QAGenerator:
    """Generates questions and answers from text content"""
    
    def __init__(self):
        """
        Initialize the question generation model
        Uses FLAN-T5-base for better Q&A generation
        """
        print("🎓 Initializing Q&A Generator...")
        
        # Use FLAN-T5-base - instruction-tuned model
        model_name = Config.QA_MODEL
        print(f"📦 Loading model: {model_name}")
        print("⚠️ This is a medium-sized model (~1GB), first download may take time...")
        
        # Load tokenizer
        print("🔤 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model on CPU
        print("🧠 Loading question generation model on CPU...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cpu"
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Q&A Generator initialized successfully!\n")
    
    def split_into_sentences(self, text):
        """
        Split text into sentences for processing
        
        Args:
            text (str): Text to split
        
        Returns:
            list: List of sentences
        """
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        return sentences
    
    def generate_question(self, context, max_questions=2):
        """
        Generate questions from context using FLAN-T5
        
        Args:
            context (str): Context paragraph
            max_questions (int): Max questions to generate
        
        Returns:
            list: List of dictionaries with question and answer
        """
        # Use FLAN-T5's instruction format
        prompt = f"Generate {max_questions} questions and answers from this text:\n\n{context[:500]}\n\nFormat: Q: [question] A: [answer]"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate Q&A
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=200,
                num_beams=4,
                early_stopping=True,
                temperature=0.7
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse Q&A pairs from result
        qa_pairs = []
        lines = result.split('\n')
        current_q = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q:'):
                current_q = line[2:].strip()
            elif line.startswith('A:') and current_q:
                answer = line[2:].strip()
                qa_pairs.append({
                    'question': current_q,
                    'answer': answer
                })
                current_q = None
        
        return qa_pairs if qa_pairs else [{'question': 'What is this text about?', 'answer': context[:100]}]
    
    def extract_key_phrases(self, text):
        """
        Extract key phrases from text to use as answers
        Simple extraction based on noun phrases
        
        Args:
            text (str): Text to extract from
        
        Returns:
            list: List of potential answer phrases
        """
        # Simple extraction: split into chunks
        sentences = self.split_into_sentences(text)
        key_phrases = []
        
        for sentence in sentences[:5]:  # Limit to first 5 sentences
            # Extract phrases (simple: split on commas and conjunctions)
            words = sentence.split()
            if len(words) > 3:
                # Take meaningful chunks
                chunk_size = min(5, len(words) - 2)
                for i in range(len(words) - chunk_size):
                    phrase = ' '.join(words[i:i+chunk_size])
                    if len(phrase) > 15:  # Minimum phrase length
                        key_phrases.append(phrase)
        
        return key_phrases[:10]  # Return top 10 phrases
    
    def generate_qa_pairs(self, text, num_questions=5):
        """
        Generate multiple question-answer pairs from text using FLAN-T5
        
        Args:
            text (str): Source text
            num_questions (int): Number of Q&A pairs to generate
        
        Returns:
            list: List of dictionaries with 'question' and 'answer'
        """
        print(f"🎯 Generating {num_questions} Q&A pairs from text...")
        print(f"📝 Text length: {len(text)} characters")
        
        qa_pairs = []
        
        # Split text into manageable chunks
        chunks = self._chunk_text(text, max_length=500)
        print(f"📦 Split into {len(chunks)} chunks")
        
        questions_per_chunk = max(1, num_questions // max(1, min(len(chunks), 5)))
        
        for chunk_idx, chunk in enumerate(chunks[:5]):  # Process first 5 chunks
            if len(qa_pairs) >= num_questions:
                break
            
            print(f"\n🔄 Processing chunk {chunk_idx + 1}...")
            
            try:
                # Generate Q&A pairs from this chunk
                chunk_qas = self.generate_question(chunk, max_questions=questions_per_chunk)
                
                for qa in chunk_qas:
                    if len(qa_pairs) >= num_questions:
                        break
                    qa_pairs.append(qa)
                    print(f"   ✅ Q{len(qa_pairs)}: {qa['question'][:60]}...")
            
            except Exception as e:
                print(f"   ⚠️ Failed to generate questions from chunk: {str(e)}")
                continue
        
        # If we didn't get enough, add simple fallback questions
        while len(qa_pairs) < min(3, num_questions):
            qa_pairs.append({
                'question': f'What is discussed in section {len(qa_pairs) + 1}?',
                'answer': chunks[len(qa_pairs)][:100] if len(qa_pairs) < len(chunks) else 'See document for details.'
            })
        
        print(f"\n✅ Generated {len(qa_pairs)} Q&A pairs!\n")
        return qa_pairs[:num_questions]
    
    def _chunk_text(self, text, max_length=400):
        """
        Split text into chunks for processing
        
        Args:
            text (str): Text to chunk
            max_length (int): Maximum chunk length
        
        Returns:
            list: List of text chunks
        """
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def generate_qa_from_pdf_text(self, pdf_text, num_questions=10):
        """
        Generate Q&A pairs directly from PDF text
        
        Args:
            pdf_text (str): Extracted PDF text
            num_questions (int): Number of questions to generate
        
        Returns:
            dict: Summary with Q&A pairs and statistics
        """
        print("📚 Generating Q&A from PDF text...\n")
        
        qa_pairs = self.generate_qa_pairs(pdf_text, num_questions=num_questions)
        
        summary = {
            'total_qa_pairs': len(qa_pairs),
            'qa_pairs': qa_pairs,
            'source_length': len(pdf_text),
            'status': 'success' if qa_pairs else 'no_questions_generated'
        }
        
        print("📊 Summary:")
        print(f"   Generated: {len(qa_pairs)} Q&A pairs")
        print(f"   Source text: {len(pdf_text)} characters\n")
        
        return summary
