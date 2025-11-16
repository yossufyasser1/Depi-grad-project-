import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
from pathlib import Path
import json
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, tesseract_cmd: str = None):
        """Initialize PDF processor."""
        if tesseract_cmd:
            pytesseract.pytesseract.pytesseract_cmd = tesseract_cmd
    
    def load_pdf_file(self, pdf_path: str) -> List:
        """Load PDF and convert to images."""
        logger.info(f'Loading PDF: {pdf_path}')
        images = convert_from_path(pdf_path)
        logger.info(f'Converted {len(images)} pages to images')
        return images
    
    def apply_preprocessing(self, image):
        """Preprocess image for OCR."""
        # Convert PIL to OpenCV format
        open_cv_image = np.array(image)
        # Convert RGB to grayscale
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        # Thresholding for better OCR
        _, thresh = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY)
        return thresh
    
    def extract_text_ocr(self, images: List) -> str:
        """Extract text from images using Tesseract."""
        full_text = ""
        for i, image in enumerate(images):
            logger.info(f'Extracting text from page {i+1}')
            preprocessed = self.apply_preprocessing(image)
            text = pytesseract.image_to_string(preprocessed, config='--psm 3')
            full_text += f"\n--- Page {i+1} ---\n" + text
        return full_text
    
    def clean_extracted_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep basic punctuation
        text = text.replace('\x00', '')
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    def save_extracted_text(self, text: str, output_path: str):
        """Save extracted text to JSON."""
        data = {
            'text': text,
            'metadata': {
                'source': 'pdf_ocr',
                'timestamp': str(Path(output_path).stat().st_mtime)
            }
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f'Saved to {output_path}')
    
    def process_pdf(self, pdf_path: str, output_path: str) -> Dict:
        """Complete PDF processing pipeline."""
        images = self.load_pdf_file(pdf_path)
        text = self.extract_text_ocr(images)
        cleaned_text = self.clean_extracted_text(text)
        chunks = self.chunk_text(cleaned_text)
        self.save_extracted_text(cleaned_text, output_path)
        
        return {
            'text': cleaned_text,
            'chunks': chunks,
            'page_count': len(images),
            'chunk_count': len(chunks)
        }

# Usage:
# processor = PDFProcessor()
# result = processor.process_pdf('input.pdf', 'output.json')
