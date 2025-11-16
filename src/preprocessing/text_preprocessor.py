import nltk
import spacy
from collections import Counter
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self, language: str = 'english'):
        """Initialize text preprocessor."""
        self.language = language
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.info('Downloading spaCy model...')
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            self.nlp = spacy.load('en_core_web_sm')
        
        from nltk.corpus import stopwords
        self.stopwords = set(stopwords.words(language))
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences."""
        sentences = nltk.sent_tokenize(text)
        return sentences
    
    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words."""
        tokens = nltk.word_tokenize(text)
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens."""
        filtered_tokens = [token for token in tokens if token.lower() not in self.stopwords]
        return filtered_tokens
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens using spaCy."""
        doc = self.nlp(' '.join(tokens))
        lemmatized = [token.lemma_ for token in doc]
        return lemmatized
    
    def normalize_text(self, text: str) -> str:
        """Normalize text: lowercase, remove punctuation."""
        text = text.lower()
        import string
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    
    def create_vocabulary(self, corpus: List[str], min_freq: int = 2, max_vocab: int = 50000) -> Dict:
        """Build vocabulary from corpus."""
        word_freq = Counter()
        for text in corpus:
            tokens = self.tokenize_words(text)
            word_freq.update(tokens)
        
        # Filter by frequency
        filtered_words = [(word, freq) for word, freq in word_freq.items() if freq >= min_freq]
        sorted_words = sorted(filtered_words, key=lambda x: x[1], reverse=True)[:max_vocab]
        vocab = {word: i+1 for i, (word, freq) in enumerate(sorted_words)}
        
        vocab['<UNK>'] = 0  # Unknown token
        vocab['<PAD>'] = len(vocab)  # Padding token
        
        return vocab
    
    def encode_text(self, text: str, vocab: Dict) -> List[int]:
        """Convert text to token IDs."""
        tokens = self.tokenize_words(text)
        token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        return token_ids
    
    def pad_sequences(self, sequences: List[List[int]], max_length: int = 512) -> List[List[int]]:
        """Pad sequences to fixed length."""
        padded = []
        pad_token = 0  # Assuming 0 is padding token
        for seq in sequences:
            if len(seq) >= max_length:
                padded.append(seq[:max_length])
            else:
                padded.append(seq + [pad_token] * (max_length - len(seq)))
        return padded
    
    def preprocess_full(self, text: str) -> Dict:
        """Full preprocessing pipeline."""
        sentences = self.tokenize_sentences(text)
        tokens = self.tokenize_words(text)
        cleaned_tokens = self.remove_stopwords(tokens)
        lemmatized = self.lemmatize(cleaned_tokens)
        normalized = self.normalize_text(text)
        
        return {
            'sentences': sentences,
            'tokens': tokens,
            'cleaned_tokens': cleaned_tokens,
            'lemmatized': lemmatized,
            'normalized': normalized
        }

# Usage:
# preprocessor = TextPreprocessor()
# result = preprocessor.preprocess_full(text)
