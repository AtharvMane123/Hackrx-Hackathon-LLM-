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
