"""
Improved LLM Reader - Local/Gemini Language Model for generating answers
Supports Ollama locally (preferred) or Gemini 2.0 as fallback
"""
import os
import re
import time
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import google.generativeai as genai
except Exception:  # Allow running without google-generativeai when using Ollama
    genai = None
from src.config import Config

logger = logging.getLogger(__name__)

class ImprovedLLMReader:
    """Enhanced LLM reader using Gemini 2.0 for better responses"""
    
    def __init__(self):
        """Initialize provider (Ollama or Gemini) with minimal configuration and history."""
        print("🤖 Initializing LLM Reader...")

        self.conversation_history = []
        self.session_id = str(uuid.uuid4())

        # Model configuration
        self.use_ollama = bool(getattr(Config, "OLLAMA_ENABLED", False))
        self.provider = "ollama" if self.use_ollama else "gemini"
        self.model_name = (
            getattr(Config, "OLLAMA_MODEL", "llama3.1") if self.use_ollama else Config.GEMINI_MODEL
        )
        self.temperature = Config.RAG_TEMPERATURE
        self.max_output_tokens = Config.RAG_MAX_OUTPUT_TOKENS
        self.top_k = 40
        self.top_p = 0.95

        # Simple system prompt
        self.system_prompt = (
            "You are a helpful study assistant. Answer based on provided context. "
            "If context is missing, be honest and helpful."
        )

        if self.use_ollama:
            # Lazy import to avoid dependency if unused
            from langchain_community.llms import Ollama
            self.model = Ollama(
                model=getattr(Config, "OLLAMA_MODEL", "llama3.1"),
                base_url=getattr(Config, "OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=self.temperature,
            )
        else:
            # Initialize Gemini API
            self._init_genai()

        print(f"✅ LLM Reader ready (provider: {self.provider})\n")
    
    def _init_genai(self) -> None:
        """Initialize Google Generative AI with enhanced configuration."""
        if genai is None:
            raise ValueError("google-generativeai not available; set OLLAMA_ENABLED=True in Config to use Ollama.")
        api_key = Config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required (or enable OLLAMA)")

        genai.configure(api_key=api_key)
        
        # Basic safety settings
        self.safety_settings = []
        
        # Generation configuration
        self.generation_config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
        }
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            system_instruction=self.system_prompt
        )
        
        # Initialize simple chat session
        self.chat = self.model.start_chat(history=[])
    
    # Removed rate limiting for simple flow
    
    # Removed caching for simplicity
    
    # Removed caching for simplicity
    
    def update_conversation_history(self, role: str, content: str) -> None:
        """Update conversation history with new message."""
        self.conversation_history.append({"role": role, "content": content})
    
    def build_prompt(self, query: str, relevant_docs: List[str]) -> str:
        """Build enhanced prompt for Gemini with relevant context."""
        # Combine relevant docs into context
        context = "\\n\\n---DOCUMENT SEPARATOR---\\n\\n".join(relevant_docs)
        
        # Include recent conversation history
        conversation_context = ""
        history_to_include = self.conversation_history[-6:] if len(self.conversation_history) > 0 else []
        
        if history_to_include:
            conversation_context = "Recent conversation context:\\n"
            for msg in history_to_include:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {msg['content']}\\n"
            conversation_context += "\\n"
        
        prompt = f"""Based on the following documentation and conversation context:

{context}

{conversation_context}Please answer the user's question:
{query}

Important guidelines:
- Be conversational and maintain context from previous messages
- Address the user naturally without mentioning "documentation" or "AI limitations"
- If the provided context doesn't contain information relevant to the question, acknowledge this naturally
- Provide helpful, accurate responses based on the available information"""
        
        return prompt
    
    def _extract_name_from_history(self) -> Optional[str]:
        """Extract user name from conversation history if available."""
        for msg in self.conversation_history:
            if msg["role"] == "user":
                content = msg["content"].lower()
                
                # Pattern: "my name is X" or "I'm X" or "call me X"
                name_patterns = [
                    r"my name is (\\w+)",
                    r"i am (\\w+)",
                    r"i'm (\\w+)",
                    r"call me (\\w+)",
                    r"name['s]* (\\w+)"
                ]
                
                for pattern in name_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        return matches[0].capitalize()
        return None
    
    def generate_answer(self, question: str, context: str, max_length: int = 500) -> str:
        """Generate enhanced answer using selected provider with context."""
        print(f"🧠 Generating enhanced answer with {self.provider}...")
        print(f"❓ Question: {question[:80]}...")
        
        # Update conversation history with user question
        self.update_conversation_history("user", question)
        
        # Handle name-related queries
        if any(phrase in question.lower() for phrase in ["my name", "told you", "remember me", "i said earlier"]):
            name = self._extract_name_from_history()
            if name:
                response = f"Yes, you mentioned your name is {name}. How can I help you today?"
                self.update_conversation_history("assistant", response)
                return response
        
        # Build prompt with context
        if context.strip():
            relevant_docs = [doc.strip() for doc in context.split("---DOCUMENT SEPARATOR---") if doc.strip()]
            content = self.build_prompt(question, relevant_docs)
        else:
            print("📭 No context available, using general knowledge...")
            content = f"""The user asked: {question}

Please provide a helpful answer. Since no specific documentation was found, you can use your general knowledge but mention that more specific information could be available if they upload relevant documents."""
        
        print(f"📝 Context length: {len(context)} characters")
        
        try:
            if self.use_ollama:
                print("🎯 Calling Ollama...")
                answer = self.model.invoke(content)
            else:
                print("🎯 Calling Gemini 2.0 API...")
                response = self.chat.send_message(content)
                answer = response.text

            self.update_conversation_history("assistant", answer)
            print(f"✅ Answer generated ({len(answer)} characters)\n")
            return answer

        except Exception as e:
            error_msg = str(e)
            print(f"❌ {self.provider.capitalize()} call failed: {error_msg}")
            
            # Check for rate limit errors
            if "429" in error_msg or "RATE_LIMIT" in error_msg or "quota" in error_msg.lower():
                fallback_response = "⚠️ I'm experiencing high demand right now. Please wait a moment and try again."
            else:
                fallback_response = f"I encountered an issue while processing your question. Please try rephrasing or try again in a moment."
            
            self.update_conversation_history("assistant", fallback_response)
            return fallback_response
    
    # Removed post-processing for simplicity
    
    def generate_simple_answer(self, question: str, max_length: int = 200) -> str:
        """Generate answer without context using general knowledge."""
        print(f"🧠 Generating answer with general knowledge via {self.provider}...")
        
        # Update conversation history
        self.update_conversation_history("user", question)
        
        prompt = f"""The user asked: {question}

Please provide a helpful answer based on your knowledge. Since no specific documents were found, mention that uploading relevant documents would allow for more specific answers."""
        
        try:
            if self.use_ollama:
                print("🎯 Calling Ollama...")
                answer = self.model.invoke(prompt)
            else:
                print("🎯 Calling Gemini API...")
                response = self.model.generate_content(prompt)
                answer = response.text

            self.update_conversation_history("assistant", answer)
            print(f"✅ Response generated\n")
            return answer
        except Exception as e:
            error_msg = str(e)
            print(f"❌ {self.provider.capitalize()} call failed: {error_msg}\n")
            
            if "429" in error_msg or "rate" in error_msg.lower():
                fallback = "⚠️ I'm experiencing high demand. Please try again in a moment or upload relevant documents for better assistance."
            else:
                fallback = f"I don't have specific information about that topic. Please upload relevant documents for more accurate assistance."
            
            self.update_conversation_history("assistant", fallback)
            return fallback
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the current conversation."""
        return {
            "session_id": self.session_id,
            "message_count": len(self.conversation_history),
            "model": self.model_name,
            "database_enabled": False
        }