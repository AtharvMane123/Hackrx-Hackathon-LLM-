import asyncio
import aiohttp
import hashlib
import io
import time
import re
from typing import Dict, List, Optional
import PyPDF2
import fitz
from concurrent.futures import ThreadPoolExecutor
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class UniversalPDFProcessor:
    def __init__(self):
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.cache = {}
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'UniversalPDF/1.0'}
        )
        logger.info("✅ Universal PDF Processor initialized")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        self.executor.shutdown(wait=False)
    
    async def process_any_pdf(self, url: str, session_id: str) -> Dict:
        """Process ANY PDF document dynamically"""
        start_time = time.time()
        
        cache_key = hashlib.md5(url.encode()).hexdigest()
        if cache_key in self.cache:
            result = self.cache[cache_key].copy()
            result.update({
                'processing_time': time.time() - start_time,
                'from_cache': True,
                'session_id': session_id
            })
            logger.info(f"⚡ Using cached PDF for {session_id}")
            return result
        
        try:
            # Step 1: Download PDF
            logger.info(f"📥 Downloading PDF: {url[:80]}...")
            pdf_content = await self._robust_pdf_download(url)
            
            # Step 2: Extract ALL text content
            logger.info(f"📖 Extracting complete text content...")
            text_result = await self._comprehensive_text_extraction(pdf_content)
            
            if not text_result['success']:
                raise Exception(f"Text extraction failed: {text_result['error']}")
            
            # Step 3: Create comprehensive chunks
            logger.info(f"🔗 Creating comprehensive chunks...")
            chunks = self._create_comprehensive_chunks(text_result['full_text'])
            
            # Step 4: Build searchable index
            logger.info(f"🔍 Building searchable index...")
            search_index = self._build_search_index(chunks)
            
            result = {
                'success': True,
                'session_id': session_id,
                'full_text': text_result['full_text'],
                'total_chunks': len(chunks),
                'chunks': chunks,
                'search_index': search_index,
                'text_length': len(text_result['full_text']),
                'pages_processed': text_result.get('pages', 0),
                'processing_time': time.time() - start_time,
                'extraction_method': text_result['method'],
                'from_cache': False
            }
            
            # Cache with memory management
            self.cache[cache_key] = result.copy()
            if len(self.cache) > 5:
                oldest = next(iter(self.cache))
                del self.cache[oldest]
            
            logger.info(f"✅ PDF processed: {len(chunks)} chunks, {len(text_result['full_text']):,} chars")
            return result
            
        except Exception as e:
            logger.error(f"❌ PDF processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'session_id': session_id
            }
    
    async def _robust_pdf_download(self, url: str) -> bytes:
        """Robust PDF download with retries"""
        last_error = None
        
        for attempt in range(3):
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        if len(content) > 500:  # Minimum viable PDF size
                            return content
                        else:
                            raise Exception(f"PDF too small: {len(content)} bytes")
                    else:
                        raise Exception(f"HTTP {response.status}: {response.reason}")
            except Exception as e:
                last_error = e
                if attempt < 2:
                    await asyncio.sleep(1)
        
        raise Exception(f"Failed to download PDF after 3 attempts: {last_error}")
    
    async def _comprehensive_text_extraction(self, pdf_content: bytes) -> Dict:
        """Comprehensive text extraction trying all methods"""
        if not pdf_content or len(pdf_content) < 100:
            return {'success': False, 'error': 'Invalid PDF content', 'full_text': ''}
        
        loop = asyncio.get_event_loop()
        
        # Method 1: Try PyMuPDF (best for structure and layout)
        try:
            logger.debug("Attempting PyMuPDF extraction...")
            result = await loop.run_in_executor(
                self.executor, self._extract_pymupdf_comprehensive, pdf_content
            )
            if result['success'] and len(result['full_text']) > 100:
                logger.info(f"✅ PyMuPDF success: {len(result['full_text']):,} chars")
                return result
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {e}")
        
        # Method 2: Try PyPDF2 (good fallback)
        try:
            logger.debug("Attempting PyPDF2 extraction...")
            result = await loop.run_in_executor(
                self.executor, self._extract_pypdf2_comprehensive, pdf_content
            )
            if result['success'] and len(result['full_text']) > 100:
                logger.info(f"✅ PyPDF2 success: {len(result['full_text']):,} chars")
                return result
        except Exception as e:
            logger.warning(f"PyPDF2 failed: {e}")
        
        return {'success': False, 'error': 'All extraction methods failed', 'full_text': ''}
    
    def _extract_pymupdf_comprehensive(self, pdf_content: bytes) -> Dict:
        """Comprehensive PyMuPDF extraction"""
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            all_text = []
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Method 1: Regular text extraction
                page_text = page.get_text()
                
                # Method 2: Try dict format for better structure
                if not page_text.strip():
                    try:
                        blocks = page.get_text("dict")
                        page_text = self._extract_from_blocks(blocks)
                    except:
                        pass
                
                # Method 3: Try HTML format as last resort
                if not page_text.strip():
                    try:
                        page_text = page.get_text("html")
                        page_text = self._clean_html_text(page_text)
                    except:
                        pass
                
                if page_text and page_text.strip():
                    # Clean and add page text
                    cleaned_text = self._clean_extracted_text(page_text)
                    if cleaned_text:
                        all_text.append(f"--- PAGE {page_num + 1} ---")
                        all_text.append(cleaned_text)
                        all_text.append("")  # Page separator
            
            doc.close()
            
            if not all_text:
                return {'success': False, 'error': 'No text extracted', 'full_text': ''}
            
            full_text = '\n'.join(all_text)
            
            return {
                'success': True,
                'full_text': full_text,
                'method': 'pymupdf_comprehensive',
                'pages': doc.page_count,
                'text_length': len(full_text)
            }
            
        except Exception as e:
            return {'success': False, 'error': f'PyMuPDF error: {str(e)}', 'full_text': ''}
    
    def _extract_from_blocks(self, blocks_dict: Dict) -> str:
        """Extract text from PyMuPDF blocks dictionary"""
        text_parts = []
        
        for block in blocks_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            line_text += text + " "
                    
                    if line_text.strip():
                        text_parts.append(line_text.strip())
        
        return "\n".join(text_parts)
    
    def _clean_html_text(self, html_text: str) -> str:
        """Clean HTML text from PyMuPDF"""
        import re
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html_text)
        
        # Clean up entities
        text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
        
        return text
    
    def _extract_pypdf2_comprehensive(self, pdf_content: bytes) -> Dict:
        """Comprehensive PyPDF2 extraction"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            reader = PyPDF2.PdfReader(pdf_file)
            all_text = []
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    
                    if page_text and page_text.strip():
                        cleaned_text = self._clean_extracted_text(page_text)
                        if cleaned_text:
                            all_text.append(f"--- PAGE {page_num + 1} ---")
                            all_text.append(cleaned_text)
                            all_text.append("")
                except Exception as e:
                    logger.warning(f"PyPDF2 page {page_num} failed: {e}")
                    continue
            
            if not all_text:
                return {'success': False, 'error': 'No text extracted', 'full_text': ''}
            
            full_text = '\n'.join(all_text)
            
            return {
                'success': True,
                'full_text': full_text,
                'method': 'pypdf2_comprehensive',
                'pages': len(reader.pages),
                'text_length': len(full_text)
            }
            
        except Exception as e:
            return {'success': False, 'error': f'PyPDF2 error: {str(e)}', 'full_text': ''}
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted text while preserving important content"""
        if not text:
            return ""
        
        # Normalize whitespace but preserve structure
        text = re.sub(r'\n\s*\n+', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        
        # Remove common PDF artifacts but keep content
        text = re.sub(r'^.*?(?:Page \d+|©|\(c\)|\d+/\d+).*?$', '', text, flags=re.MULTILINE)
        
        # Split into lines and clean each
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 2:  # Keep meaningful lines
                # Remove excessive punctuation but keep content
                line = re.sub(r'\.{3,}', '...', line)
                line = re.sub(r'-{3,}', '---', line)
                
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _create_comprehensive_chunks(self, full_text: str) -> List[Dict]:
        """Create comprehensive, overlapping chunks from full text"""
        if not full_text or len(full_text.strip()) < 50:
            return []
        
        chunks = []
        chunk_size = settings.CHUNK_SIZE
        overlap = settings.CHUNK_OVERLAP
        
        # Split text into sentences for better chunking
        sentences = re.split(r'[.!?]+', full_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            # Fallback to character-based chunking
            for i in range(0, len(full_text), chunk_size - overlap):
                chunk_text = full_text[i:i + chunk_size]
                if len(chunk_text.strip()) > 50:
                    chunks.append({
                        'chunk_id': f"char_chunk_{len(chunks)}",
                        'content': chunk_text.strip(),
                        'char_start': i,
                        'char_end': i + len(chunk_text),
                        'chunk_type': 'character_based'
                    })
            return chunks
        
        # Sentence-aware chunking
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                # Save current chunk
                if current_chunk.strip():
                    chunks.append({
                        'chunk_id': f"sentence_chunk_{len(chunks)}",
                        'content': current_chunk.strip(),
                        'sentences': current_sentences.copy(),
                        'chunk_type': 'sentence_based'
                    })
                
                # Start new chunk with overlap
                if len(current_sentences) > 1:
                    # Keep last sentence for overlap
                    current_sentences = [current_sentences[-1], sentence]
                    current_chunk = current_sentences[0] + ". " + sentence + "."
                else:
                    current_sentences = [sentence]
                    current_chunk = sentence + "."
            else:
                current_sentences.append(sentence)
                if current_chunk:
                    current_chunk += " " + sentence + "."
                else:
                    current_chunk = sentence + "."
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'chunk_id': f"sentence_chunk_{len(chunks)}",
                'content': current_chunk.strip(),
                'sentences': current_sentences,
                'chunk_type': 'sentence_based'
            })
        
        return chunks
    
    def _build_search_index(self, chunks: List[Dict]) -> Dict:
        """Build searchable index from chunks"""
        search_index = {
            'word_to_chunks': {},
            'phrase_to_chunks': {},
            'number_to_chunks': {}
        }
        
        for i, chunk in enumerate(chunks):
            content = chunk.get('content', '').lower()
            
            # Index individual words (4+ characters)
            words = re.findall(r'\b\w{4,}\b', content)
            for word in set(words):  # Use set to avoid duplicates
                if word not in search_index['word_to_chunks']:
                    search_index['word_to_chunks'][word] = []
                search_index['word_to_chunks'][word].append(i)
            
            # Index common phrases (2-3 words)
            word_list = content.split()
            for j in range(len(word_list) - 1):
                phrase = f"{word_list[j]} {word_list[j+1]}"
                if len(phrase) > 6:  # Meaningful phrases only
                    if phrase not in search_index['phrase_to_chunks']:
                        search_index['phrase_to_chunks'][phrase] = []
                    search_index['phrase_to_chunks'][phrase].append(i)
            
            # Index numbers
            numbers = re.findall(r'\b\d+\b', content)
            for number in set(numbers):
                if number not in search_index['number_to_chunks']:
                    search_index['number_to_chunks'][number] = []
                search_index['number_to_chunks'][number].append(i)
        
        return search_index
