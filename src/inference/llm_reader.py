from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
import torch
from typing import List, Dict, Optional, Iterator
from pathlib import Path
import logging
from threading import Thread

# Import project config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMReader:
    """LLM-based answer generator with RAG support."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        use_quantization: bool = False,
        config: Optional[Config] = None
    ):
        """Initialize LLM for answer generation."""
        self.config = config or Config()
        
        # Use config model if not specified
        if model_name is None:
            model_name = self.config.LLM_MODEL_NAME
        
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_quantization = use_quantization
        
        logger.info(f'Loading LLM: {model_name}')
        logger.info(f'Device: {self.device}, Quantization: {use_quantization}')
        
        # Load model with optional quantization
        if use_quantization and torch.cuda.is_available():
            self.model = self._load_quantized_model(model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f'Model loaded with {total_params:,} parameters')
    
    def _load_quantized_model(self, model_name: str):
        """Load model with 4-bit quantization for efficiency."""
        logger.info('Loading model with 4-bit quantization...')
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map='auto',
            torch_dtype=torch.bfloat16
        )
        
        logger.info('Quantized model loaded successfully')
        return model
    
    def create_rag_prompt(
        self,
        question: str,
        context_docs: List[str],
        template: str = 'standard',
        max_context_length: int = 2048
    ) -> str:
        """Create RAG prompt with context documents."""
        
        if template == 'standard':
            # Standard QA format
            context_text = '\n\n'.join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context_docs)])
            
            # Truncate context if too long
            if len(context_text) > max_context_length:
                context_text = context_text[:max_context_length] + "..."
            
            prompt = f"""You are a helpful AI study assistant. Answer the following question based on the provided context documents. If the answer is not in the context, say so.

Context:
{context_text}

Question: {question}

Answer:"""
        
        elif template == 'concise':
            # Concise format
            context_text = '\n'.join([f"- {doc[:200]}..." for doc in context_docs])
            prompt = f"Context: {context_text}\n\nQuestion: {question}\n\nAnswer:"
        
        elif template == 'instructional':
            # Instruction-following format (better for instruction-tuned models)
            context_text = '\n'.join(context_docs)
            prompt = f"""### Instruction:
Use the following context to answer the question. Be concise and accurate.

### Context:
{context_text}

### Question:
{question}

### Answer:"""
        
        else:
            # Simple format
            context_text = ' '.join(context_docs)
            prompt = f"Context: {context_text}\n\nQ: {question}\nA:"
        
        return prompt
    
    def generate_answer(
        self,
        question: str,
        context: Optional[str] = None,
        context_docs: Optional[List[str]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        template: str = 'standard'
    ) -> str:
        """Generate answer from question and context."""
        
        # Prepare context
        if context_docs:
            contexts = context_docs
        elif context:
            contexts = [context]
        else:
            contexts = []
        
        # Create prompt
        if contexts:
            prompt = self.create_rag_prompt(question, contexts, template=template)
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        
        logger.info(f'Generating answer for: "{question[:50]}..."')
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=self.config.MAX_LENGTH
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (remove prompt)
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text[len(prompt):].strip()
        
        logger.info('Answer generated successfully')
        return answer
    
    def generate_with_retrieval(
        self,
        question: str,
        retrieved_docs: List[Dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7
    ) -> Dict:
        """Generate answer using retrieved documents."""
        
        # Extract document texts
        context_docs = [doc['text'] for doc in retrieved_docs]
        
        # Generate answer
        answer = self.generate_answer(
            question=question,
            context_docs=context_docs,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        # Return structured result
        result = {
            'question': question,
            'answer': answer,
            'sources': retrieved_docs,
            'num_sources': len(retrieved_docs)
        }
        
        return result
    
    def stream_generation(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7
    ) -> Iterator[str]:
        """Stream token generation for real-time responses."""
        logger.info('Starting streaming generation...')
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=self.config.MAX_LENGTH
        ).to(self.device)
        
        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Generation kwargs
        generation_kwargs = {
            **inputs,
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': 0.95,
            'do_sample': True,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'streamer': streamer
        }
        
        # Start generation in separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield tokens as they're generated
        for token in streamer:
            yield token
    
    def batch_generate(
        self,
        questions: List[str],
        contexts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7
    ) -> List[str]:
        """Generate answers for multiple questions in batch."""
        logger.info(f'Batch generating answers for {len(questions)} questions')
        
        # Create prompts
        prompts = [
            self.create_rag_prompt(q, [c], template='concise')
            for q, c in zip(questions, contexts)
        ]
        
        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.MAX_LENGTH
        ).to(self.device)
        
        # Generate batch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode all outputs
        answers = []
        for i, output in enumerate(outputs):
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Extract answer part
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text[len(prompts[i]):].strip()
            answers.append(answer)
        
        logger.info('Batch generation completed')
        return answers
    
    def get_model_info(self) -> Dict:
        """Get model information and configuration."""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'quantized': self.use_quantization,
            'vocab_size': self.model.config.vocab_size,
            'max_position_embeddings': getattr(self.model.config, 'max_position_embeddings', 'N/A'),
            'num_parameters': sum(p.numel() for p in self.model.parameters())
        }

# Usage:
# # Initialize with default LLM from config
# reader = LLMReader()
#
# # Generate answer with context
# answer = reader.generate_answer(
#     question="What is machine learning?",
#     context="Machine learning is a subset of AI..."
# )
#
# # Generate with retrieved documents
# from src.inference.rag_retriever import RAGRetriever
# docs = retriever.retrieve_documents("What is ML?")
# result = reader.generate_with_retrieval("What is ML?", docs)
#
# # Stream generation
# for token in reader.stream_generation("Explain AI:", max_new_tokens=100):
#     print(token, end='', flush=True)
