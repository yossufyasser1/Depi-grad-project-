"""
PDF Processor - Extracts and chunks text from PDF files
Simple text extraction without OCR
"""
from pypdf import PdfReader
import os


class PDFProcessor:
    """Handles PDF text extraction and chunking"""
    
    def __init__(self, chunk_size=500):
        """
        Initialize PDF processor
        
        Args:
            chunk_size (int): Size of text chunks in characters (default: 500)
        """
        self.chunk_size = chunk_size
        print(f"📄 PDF Processor initialized (chunk size: {chunk_size} chars)\n")
    
    def extract_text(self, pdf_path):
        """
        Extract all text from a PDF file
        
        Args:
            pdf_path (str): Path to the PDF file
        
        Returns:
            str: Extracted text from all pages
        """
        print(f"📖 Extracting text from: {os.path.basename(pdf_path)}")
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            print(f"❌ Error: File not found - {pdf_path}\n")
            return ""
        
        try:
            # Open and read PDF
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            print(f"📊 Total pages: {total_pages}")
            
            # Extract text from all pages
            all_text = ""
            for page_num, page in enumerate(reader.pages, 1):
                print(f"   Processing page {page_num}/{total_pages}...")
                page_text = page.extract_text()
                all_text += page_text + "\n"
            
            # Clean up extra whitespace
            all_text = all_text.strip()
            
            print(f"✅ Extraction complete: {len(all_text)} characters\n")
            return all_text
            
        except Exception as e:
            print(f"❌ Error extracting text: {str(e)}\n")
            return ""
    
    def chunk_text(self, text):
        """
        Split text into chunks of specified size
        
        Args:
            text (str): Text to split into chunks
        
        Returns:
            list: List of text chunks
        """
        print(f"✂️ Splitting text into chunks of {self.chunk_size} characters...")
        
        if not text:
            print("⚠️ No text to chunk\n")
            return []
        
        chunks = []
        
        # Split text into chunks with overlap for context continuity
        overlap = 50  # Characters to overlap between chunks
        start = 0
        
        while start < len(text):
            # Get chunk from start to start + chunk_size
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk.strip())
            
            # Move start position (with overlap)
            start = end - overlap
        
        print(f"✅ Created {len(chunks)} chunks\n")
        return chunks
    
    def process_pdf(self, pdf_path):
        """
        Complete pipeline: Extract text from PDF and split into chunks
        
        Args:
            pdf_path (str): Path to the PDF file
        
        Returns:
            tuple: (all_text, chunks) - Full text and list of chunks
        """
        print("🚀 Starting PDF processing pipeline...\n")
        
        # Extract text
        text = self.extract_text(pdf_path)
        
        if not text:
            print("⚠️ No text extracted from PDF\n")
            return "", []
        
        # Create chunks
        chunks = self.chunk_text(text)
        
        print(f"✅ Pipeline complete!")
        print(f"   📝 Total text: {len(text)} characters")
        print(f"   📦 Total chunks: {len(chunks)}\n")
        
        return text, chunks
    
    def process_multiple_pdfs(self, pdf_paths):
        """
        Process multiple PDF files
        
        Args:
            pdf_paths (list): List of PDF file paths
        
        Returns:
            dict: Dictionary with filenames as keys and (text, chunks) as values
        """
        print(f"📚 Processing {len(pdf_paths)} PDF files...\n")
        
        results = {}
        
        for i, pdf_path in enumerate(pdf_paths, 1):
            print(f"--- File {i}/{len(pdf_paths)} ---")
            filename = os.path.basename(pdf_path)
            text, chunks = self.process_pdf(pdf_path)
            results[filename] = {
                'text': text,
                'chunks': chunks,
                'chunk_count': len(chunks),
                'char_count': len(text)
            }
        
        print(f"✅ All PDFs processed!\n")
        return results
