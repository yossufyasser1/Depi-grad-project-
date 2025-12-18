"""
Improved LLM Reader - Local Ollama Language Model for generating answers
"""
import os
import uuid
import logging
from src.config import Config

logger = logging.getLogger(__name__)

class ImprovedLLMReader:
    # LLM reader using Ollama for local generation
    
    def __init__(self):
        # Initialize Ollama with conversation history
        print("Initializing LLM Reader...")

        self.conversation_history = []
        self.session_id = str(uuid.uuid4())

        # Model configuration
        self.model_name = Config.OLLAMA_MODEL
        self.temperature = Config.RAG_TEMPERATURE

        # System prompt
        self.system_prompt = (
            "You are a helpful study assistant. Answer based on provided context. "
            "If context is missing, be honest and helpful."
        )

        # Initialize Ollama
        from langchain_community.llms import Ollama
        self.model = Ollama(
            model=self.model_name,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=self.temperature,
        )

        print(f" LLM Reader ready (model: {self.model_name})\n")
    
    # Ollama-only implementation - no additional provider setup needed
    
    def update_conversation_history(self, role, content):
        # Update conversation history with new message
        self.conversation_history.append({"role": role, "content": content})
    
    def build_prompt(self, query, relevant_docs):
        # Build enhanced prompt with relevant context
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
    

    def generate_answer(self, question, context, max_length=500):
        # Generate answer using Ollama with context
        print(f"🧠 Generating answer with Ollama...")
        print(f"❓ Question: {question[:80]}...")
        
        # Update conversation history with user question
        self.update_conversation_history("user", question)
        
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
            print("🎯 Calling Ollama...")
            answer = self.model.invoke(content)
            self.update_conversation_history("assistant", answer)
            print(f"✅ Answer generated ({len(answer)} characters)\n")
            return answer

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Ollama call failed: {error_msg}")
            fallback_response = "I encountered an issue while processing your question. Please ensure Ollama is running and try again."
            self.update_conversation_history("assistant", fallback_response)
            return fallback_response
    
    # Removed post-processing for simplicity
    
    def generate_simple_answer(self, question, max_length=200):
        # Generate answer without context using Ollama
        print(f"🧠 Generating answer with Ollama...")
        
        # Update conversation history
        self.update_conversation_history("user", question)
        
        prompt = f"""The user asked: {question}

Please provide a helpful answer based on your knowledge. Since no specific documents were found, mention that uploading relevant documents would allow for more specific answers."""
        
        try:
            print("🎯 Calling Ollama...")
            answer = self.model.invoke(prompt)
            self.update_conversation_history("assistant", answer)
            print(f"✅ Response generated\n")
            return answer
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Ollama call failed: {error_msg}\n")
            fallback = "I don't have specific information about that topic. Please ensure Ollama is running and upload relevant documents for more accurate assistance."
            self.update_conversation_history("assistant", fallback)
            return fallback
    
    def get_conversation_stats(self):
        # Get statistics about the current conversation
        return {
            "session_id": self.session_id,
            "message_count": len(self.conversation_history),
            "model": self.model_name,
            "database_enabled": False
        }