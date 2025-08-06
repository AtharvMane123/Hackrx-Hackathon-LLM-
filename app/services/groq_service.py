import asyncio
import time
import logging
import re
from typing import Dict, List, Optional
from groq import AsyncGroq
from app.config import settings

logger = logging.getLogger(__name__)

class GroqService:
    def __init__(self):
        self.client = None
        
    async def __aenter__(self):
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY required")
        
        try:
            self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
            logger.info("✅ Groq client initialized")
            return self
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            try:
                await self.client.aclose()
            except Exception as e:
                logger.warning(f"Groq client close error (safe to ignore): {e}")
    
    async def generate_answer(self, question: str, context: str) -> Dict:
        """Generate clean, complete answers like the expected format"""
        start_time = time.time()
        
        if not context or len(context) < 30:
            return {
                'success': False,
                'error': 'Insufficient context',
                'processing_time': time.time() - start_time
            }
        
        # Limit context for optimal processing
        if len(context) > 1800:
            context = context[:1800]
        
        # Enhanced prompt for complete, clean answers
        prompt = f"""Based on the insurance policy document below, provide a complete, professional answer to the question.

POLICY DOCUMENT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Provide a complete sentence that fully answers the question
- Include specific details (numbers, timeframes, conditions) from the document
- Write in clear, professional language
- Start with direct information (e.g., "A grace period of...", "There is a waiting period of...", "Yes, the policy covers...")
- Make the answer self-contained and easy to understand
- Extract the actual policy details, not just definitions

COMPLETE ANSWER:"""
        
        try:
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an insurance policy analyst who provides complete, clear answers with specific details from policy documents."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-70b-versatile",  # Use more accurate model
                temperature=0.2,  # Slightly higher for more natural language
                max_tokens=150,   # Allow for complete sentences
                top_p=0.9
            )
            
            content = response.choices[0].message.content
            
            if content and len(content.strip()) > 10:
                # Clean and ensure complete answer
                cleaned_answer = self._clean_and_complete_answer(content)
                
                return {
                    'success': True,
                    'answer': cleaned_answer,
                    'processing_time': time.time() - start_time,
                    'tokens_used': response.usage.total_tokens if response.usage else 0
                }
        
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
        
        return {
            'success': False,
            'error': 'Groq generation failed',
            'processing_time': time.time() - start_time
        }
    
    def _clean_and_complete_answer(self, response: str) -> str:
        """Clean response to match expected format"""
        if not response:
            return "No response generated."
        
        # Remove technical prefixes
        prefixes_to_remove = [
            r'^(?:the\s+)?(?:policy\s+)?(?:states?\s+that\s+)?',
            r'^(?:according\s+to\s+(?:the\s+)?(?:document|policy)\s*,?\s*)?',
            r'^(?:based\s+on\s+(?:the\s+)?(?:document|policy)\s*,?\s*)?',
            r'^(?:the\s+document\s+(?:states?|indicates?|shows?)\s+(?:that\s+)?)?',
            r'^(?:as\s+per\s+(?:the\s+)?(?:document|policy)\s*,?\s*)?'
        ]
        
        cleaned = response.strip()
        for prefix_pattern in prefixes_to_remove:
            cleaned = re.sub(prefix_pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Ensure proper capitalization
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        # Clean up common issues
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces
        cleaned = re.sub(r'^\s*["\']', '', cleaned)  # Leading quotes
        cleaned = re.sub(r'["\']?\s*$', '', cleaned)  # Trailing quotes
        
        # Ensure proper ending
        if not cleaned.endswith(('.', '!', '?')):
            cleaned += '.'
        
        return cleaned
    
    def _clean_groq_response(self, response: str) -> str:
        """Clean Groq response to ensure single sentence"""
        if not response:
            return "No response generated."
        
        # Remove common prefixes
        prefixes_to_remove = [
            r'^(?:the\s+)?(?:policy\s+)?(?:states?\s+that\s+)?',
            r'^(?:according\s+to\s+(?:the\s+)?(?:document|policy)\s*,?\s*)?',
            r'^(?:based\s+on\s+(?:the\s+)?(?:document|policy)\s*,?\s*)?',
            r'^(?:the\s+document\s+(?:states?|indicates?|shows?)\s+(?:that\s+)?)?'
        ]
        
        cleaned = response.strip()
        for prefix_pattern in prefixes_to_remove:
            cleaned = re.sub(prefix_pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Get first sentence
        sentences = re.split(r'[.!?]+', cleaned)
        first_sentence = sentences[0].strip()
        
        if len(first_sentence) > 15:
            # Ensure proper capitalization
            if first_sentence and first_sentence[0].islower():
                first_sentence = first_sentence[0].upper() + first_sentence[1:]
            
            # Ensure proper ending
            if not first_sentence.endswith(('.', '!', '?')):
                first_sentence += '.'
            
            return first_sentence
        
        # Return cleaned original if first sentence too short
        if not cleaned.endswith(('.', '!', '?')):
            cleaned += '.'
        
        return cleaned
