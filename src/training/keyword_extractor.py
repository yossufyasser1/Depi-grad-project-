from collections import Counter
import re
from typing import List, Tuple
import nltk

class KeywordExtractor:
    def __init__(self, stop_words: List[str] = None, language: str = 'english'):
        """Initialize keyword extractor."""
        if stop_words is None:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words(language))
        else:
            self.stop_words = set(stop_words)
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Extract top N keywords using RAKE algorithm."""
        # Extract candidate keywords (phrases separated by stop words)
        sentences = nltk.sent_tokenize(text)
        candidates = []
        
        for sentence in sentences:
            words = nltk.word_tokenize(sentence.lower())
            phrase = []
            
            for word in words:
                if word not in self.stop_words and word.isalnum():
                    phrase.append(word)
                elif phrase:
                    candidates.append(' '.join(phrase))
                    phrase = []
            
            if phrase:
                candidates.append(' '.join(phrase))
        
        # Score candidates
        word_freq = Counter()
        word_degree = Counter()
        
        for candidate in candidates:
            words = candidate.split()
            for word in words:
                word_freq[word] += 1
                word_degree[word] += len(words)
        
        # Calculate scores
        keyword_scores = {}
        for candidate in set(candidates):
            words = candidate.split()
            score = 0
            for word in words:
                score += word_degree[word] / word_freq[word]
            keyword_scores[candidate] = score
        
        # Return top keywords
        top_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return top_keywords

# Usage:
# extractor = KeywordExtractor()
# keywords = extractor.extract_keywords(text, top_n=10)
