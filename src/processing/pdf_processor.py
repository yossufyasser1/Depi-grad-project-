"""
PDF Processor - Extracts and chunks text from PDF files
Simple text extraction without OCR
"""
from pypdf import PdfReader
import os


class PDFProcessor:
    """Handles simple PDF text extraction."""
    
    def __init__(self):
        """Initialize a minimal PDF processor."""
        print("📄 PDF Processor initialized (extraction only)\n")
    
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
    
    # Removed chunking to keep RAG responsibilities in ImprovedRAGRetriever
    
    def process_pdf(self, pdf_path):
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
        
        Returns:
            str: Extracted full text
        """
        print("🚀 Starting PDF extraction pipeline...\n")
        text = self.extract_text(pdf_path)
        if not text:
            print("⚠️ No text extracted from PDF\n")
            return ""
        print(f"✅ Extraction complete! 📝 Total text: {len(text)} characters\n")
        return text
    
    def process_multiple_pdfs(self, pdf_paths):
        """
        Extract text from multiple PDF files (no chunking)
        
        Args:
            pdf_paths (list): List of PDF file paths
        
        Returns:
            dict: Dictionary with filenames as keys and extracted text/length
        """
        print(f"📚 Processing {len(pdf_paths)} PDF files...\n")
        
        results = {}
        
        for i, pdf_path in enumerate(pdf_paths, 1):
            print(f"--- File {i}/{len(pdf_paths)} ---")
            filename = os.path.basename(pdf_path)
            text = self.process_pdf(pdf_path)
            results[filename] = {
                'text': text,
                'char_count': len(text)
            }
        
        print(f"✅ All PDFs processed!\n")
        return results
