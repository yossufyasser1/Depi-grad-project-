import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from typing import List
import numpy as np

class TextRankSummarizer:
    def __init__(self, num_sentences: int = 3, language: str = 'english'):
        """Initialize TextRank summarizer."""
        self.num_sentences = num_sentences
        self.language = language
    
    def _score_sentences(self, sentences: List[str]) -> np.ndarray:
        """Score sentences using TF-IDF and PageRank."""
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(stop_words=self.language)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Cosine similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Build graph
        nx_graph = nx.from_numpy_array(similarity_matrix)
        
        # PageRank
        scores = nx.pagerank(nx_graph)
        
        return np.array([scores[i] for i in range(len(sentences))])
    
    def summarize(self, text: str, num_sentences: int = None) -> str:
        """Extract key sentences using TextRank."""
        if num_sentences is None:
            num_sentences = self.num_sentences
        
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        scores = self._score_sentences(sentences)
        
        top_indices = np.argsort(scores)[-num_sentences:]
        top_indices.sort()
        
        summary = ' '.join([sentences[i] for i in top_indices])
        
        return summary

# Usage:
# summarizer = TextRankSummarizer()
# summary = summarizer.summarize(long_text)
