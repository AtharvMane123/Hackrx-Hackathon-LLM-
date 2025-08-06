import re
import logging
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

class DynamicAnswerExtractor:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),  # Include phrases
            lowercase=True
        )
    
    def extract_answer(self, question: str, full_text: str, chunks: List[Dict], search_index: Dict) -> str:
        """100% Dynamic answer extraction from ANY document"""
        
        if not question or not full_text:
            return "Insufficient information to process the question."
        
        # Method 1: Fast index-based search
        relevant_chunks = self._search_using_index(question, chunks, search_index)
        
        if relevant_chunks:
            answer = self._extract_from_chunks(question, relevant_chunks)
            if self._is_good_answer(answer):
                return self._format_final_answer(answer)
        
        # Method 2: TF-IDF similarity search
        similar_chunks = self._tfidf_similarity_search(question, chunks)
        
        if similar_chunks:
            answer = self._extract_from_chunks(question, similar_chunks)
            if self._is_good_answer(answer):
                return self._format_final_answer(answer)
        
        # Method 3: Direct text search
        answer = self._direct_text_search(question, full_text)
        if self._is_good_answer(answer):
            return self._format_final_answer(answer)
        
        # Method 4: Keyword expansion search
        answer = self._keyword_expansion_search(question, full_text)
        if self._is_good_answer(answer):
            return self._format_final_answer(answer)
        
        return "The specific information requested was not found in the document."
    
    def _search_using_index(self, question: str, chunks: List[Dict], search_index: Dict) -> List[Dict]:
        """Fast search using pre-built index"""
        question_lower = question.lower()
        
        # Extract question terms
        question_words = set(re.findall(r'\b\w{4,}\b', question_lower))
        question_phrases = self._extract_phrases(question_lower)
        question_numbers = set(re.findall(r'\b\d+\b', question_lower))
        
        chunk_scores = {}
        
        # Score chunks based on word matches
        for word in question_words:
            if word in search_index['word_to_chunks']:
                for chunk_idx in search_index['word_to_chunks'][word]:
                    chunk_scores[chunk_idx] = chunk_scores.get(chunk_idx, 0) + 1
        
        # Score chunks based on phrase matches (higher weight)
        for phrase in question_phrases:
            if phrase in search_index['phrase_to_chunks']:
                for chunk_idx in search_index['phrase_to_chunks'][phrase]:
                    chunk_scores[chunk_idx] = chunk_scores.get(chunk_idx, 0) + 3
        
        # Score chunks based on number matches (very high weight for policy docs)
        for number in question_numbers:
            if number in search_index['number_to_chunks']:
                for chunk_idx in search_index['number_to_chunks'][number]:
                    chunk_scores[chunk_idx] = chunk_scores.get(chunk_idx, 0) + 5
        
        # Get top scoring chunks
        if not chunk_scores:
            return []
        
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        relevant_chunks = []
        for chunk_idx, score in sorted_chunks[:5]:  # Top 5 chunks
            chunk = chunks[chunk_idx].copy()
            chunk['relevance_score'] = score
            relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    def _extract_phrases(self, text: str) -> List[str]:
        """Extract meaningful phrases from text"""
        words = text.split()
        phrases = []
        
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            if len(phrase) > 6:  # Meaningful length
                phrases.append(phrase)
        
        return phrases
    
    def _tfidf_similarity_search(self, question: str, chunks: List[Dict]) -> List[Dict]:
        """TF-IDF based similarity search"""
        if not chunks:
            return []
        
        try:
            # Prepare texts
            texts = [question] + [chunk.get('content', '') for chunk in chunks]
            
            # Fit TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Calculate similarities
            question_vector = tfidf_matrix[0:1]
            chunk_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(question_vector, chunk_vectors)[0]
            
            # Get top similar chunks
            chunk_similarities = list(enumerate(similarities))
            chunk_similarities.sort(key=lambda x: x[1], reverse=True)
            
            relevant_chunks = []
            for chunk_idx, similarity in chunk_similarities[:5]:
                if similarity > 0.1:  # Minimum similarity threshold
                    chunk = chunks[chunk_idx].copy()
                    chunk['relevance_score'] = float(similarity)
                    relevant_chunks.append(chunk)
            
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"TF-IDF search failed: {e}")
            return []
    
    def _extract_from_chunks(self, question: str, chunks: List[Dict]) -> str:
        """Extract best answer from relevant chunks"""
        if not chunks:
            return ""
        
        # Combine top chunks
        combined_text = ""
        for chunk in chunks[:3]:  # Use top 3 chunks
            content = chunk.get('content', '')
            if content:
                combined_text += content + " "
        
        if not combined_text:
            return ""
        
        # Find most relevant sentences in the combined text
        return self._find_best_sentence(question, combined_text)
    
    def _find_best_sentence(self, question: str, text: str) -> str:
        """Find the best sentence that answers the question"""
        if not text:
            return ""
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return ""
        
        # Extract question keywords
        question_words = set(re.findall(r'\b\w{3,}\b', question.lower()))
        
        # Score sentences
        scored_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Count keyword matches
            word_matches = sum(1 for word in question_words if word in sentence_lower)
            
            # Bonus for numbers (important in policy documents)
            number_bonus = len(re.findall(r'\b\d+\b', sentence)) * 0.5
            
            # Bonus for policy-related terms
            policy_terms = ['policy', 'coverage', 'benefit', 'premium', 'insured', 'claim', 'period', 'waiting', 'grace']
            policy_bonus = sum(0.3 for term in policy_terms if term in sentence_lower)
            
            # Bonus for sentence length (more complete answers)
            length_bonus = min(1.0, len(sentence) / 100)
            
            total_score = word_matches + number_bonus + policy_bonus + length_bonus
            
            if total_score > 1:  # Minimum relevance threshold
                scored_sentences.append({
                    'sentence': sentence,
                    'score': total_score,
                    'word_matches': word_matches
                })
        
        if not scored_sentences:
            return ""
        
        # Sort by score and return best
        scored_sentences.sort(key=lambda x: (x['word_matches'], x['score']), reverse=True)
        
        best_sentence = scored_sentences[0]['sentence']
        return best_sentence
    
    def _direct_text_search(self, question: str, full_text: str) -> str:
        """Direct search in full text"""
        question_words = re.findall(r'\b\w{3,}\b', question.lower())
        
        if not question_words:
            return ""
        
        # Create search patterns for different combinations of question words
        search_patterns = []
        
        # All words pattern
        if len(question_words) > 1:
            all_words_pattern = r'[^.]*?' + r'[^.]*?'.join(re.escape(word) for word in question_words) + r'[^.]*?(?:\.|$)'
            search_patterns.append(all_words_pattern)
        
        # Individual word patterns with context
        for word in question_words:
            pattern = r'[^.]*?' + re.escape(word) + r'[^.]{0,200}?(?:\.|$)'
            search_patterns.append(pattern)
        
        best_match = ""
        best_score = 0
        
        for pattern in search_patterns:
            matches = list(re.finditer(pattern, full_text, re.IGNORECASE | re.DOTALL))
            
            for match in matches:
                matched_text = match.group(0).strip()
                
                if len(matched_text) < 20:  # Skip very short matches
                    continue
                
                # Score based on question word density
                matched_lower = matched_text.lower()
                word_count = sum(1 for word in question_words if word in matched_lower)
                
                # Bonus for numbers
                number_count = len(re.findall(r'\b\d+\b', matched_text))
                
                # Total score
                score = word_count * 2 + number_count + (len(matched_text) / 100)
                
                if score > best_score:
                    best_score = score
                    best_match = matched_text
        
        return best_match
    
    def _keyword_expansion_search(self, question: str, full_text: str) -> str:
        """Search using expanded keywords"""
        
        # Define keyword expansions for common terms
        expansions = {
            'grace': ['grace', 'period', 'premium', 'payment', 'due'],
            'waiting': ['waiting', 'period', 'months', 'years', 'coverage'],
            'maternity': ['maternity', 'pregnancy', 'childbirth', 'delivery', 'female'],
            'cataract': ['cataract', 'surgery', 'eye', 'operation'],
            'organ': ['organ', 'donor', 'transplant', 'medical', 'expenses'],
            'hospital': ['hospital', 'institution', 'beds', 'medical', 'facility'],
            'room': ['room', 'rent', 'charges', 'icu', 'limit']
        }
        
        question_lower = question.lower()
        expanded_keywords = set(re.findall(r'\b\w{3,}\b', question_lower))
        
        # Expand keywords
        for key, expansion in expansions.items():
            if key in question_lower:
                expanded_keywords.update(expansion)
        
        # Search for sentences containing expanded keywords
        sentences = re.split(r'[.!?]+', full_text)
        
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            if len(sentence.strip()) < 30:
                continue
                
            sentence_lower = sentence.lower()
            
            # Count expanded keyword matches
            matches = sum(1 for keyword in expanded_keywords if keyword in sentence_lower)
            
            if matches > best_score:
                best_score = matches
                best_sentence = sentence.strip()
        
        return best_sentence
    
    def _is_good_answer(self, answer: str) -> bool:
        """Check if the extracted answer is meaningful"""
        if not answer or len(answer.strip()) < 15:
            return False
        
        # Check for generic non-answers
        generic_phrases = [
            'not found', 'not available', 'not specified', 'not mentioned',
            'unable to determine', 'insufficient information', 'no information'
        ]
        
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in generic_phrases):
            return False
        
        # Must contain at least some meaningful content
        words = re.findall(r'\b\w+\b', answer)
        if len(words) < 5:
            return False
        
        return True
    
    def _format_final_answer(self, answer: str) -> str:
        """Format clean, short answers"""
        if not answer:
            return ""
        
        # Remove document artifacts and codes
        answer = re.sub(r'CBD-\d+[^,]*,?\s*', '', answer)  # Remove CBD codes
        answer = re.sub(r'UIN:\s*[A-Z0-9]+\s*', '', answer)  # Remove UIN codes
        answer = re.sub(r'Table of Benefits[^.]*', '', answer)  # Remove table references
        answer = re.sub(r'Features Plans PLAN [ABC][^.]*', '', answer)  # Remove plan tables
        answer = re.sub(r'Sum insured.*?Lac', '', answer)  # Remove sum insured details
        
        # Clean whitespace and artifacts
        answer = re.sub(r'^\s*[-\d.]+\s*', '', answer)  # Remove leading numbers
        answer = re.sub(r'Page \d+', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Get first meaningful sentence
        sentences = re.split(r'[.!?]+', answer)
        if sentences:
            first_sentence = sentences[0].strip()
            
            # Skip if too technical or contains artifacts
            if any(term in first_sentence for term in ['CBD-', 'UIN:', 'Table of', 'PLAN A', 'PLAN B']):
                # Try next sentence
                if len(sentences) > 1:
                    first_sentence = sentences[1].strip()
            
            if len(first_sentence) > 15:
                # Ensure proper format
                if first_sentence[0].islower():
                    first_sentence = first_sentence[0].upper() + first_sentence[1:]
                
                if not first_sentence.endswith('.'):
                    first_sentence += '.'
                    
                return first_sentence
        
        return "Information not found in a readable format."
