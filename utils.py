"""Text preprocessing and the term-document matching shared by intent matching and Q&A."""
import string

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

# The lemmatiser needs WordNet; download it once if it's missing
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

# Punctuation to strip, including curly quotes and dashes
_STRIP_PUNCTUATION = str.maketrans("", "", string.punctuation + "‘’“”–—")


class Utils:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text):
        """Lowercase, remove punctuation, split into words and lemmatise (stopwords are kept)."""
        words = str(text).translate(_STRIP_PUNCTUATION).lower().split()
        return [self.lemmatizer.lemmatize(word) for word in words]

    def build_vocabulary(self, texts):
        """Sorted list of every distinct term in the given texts."""
        vocab = set()
        for text in texts:
            vocab.update(self.preprocess(text))
        return sorted(vocab)

    def build_term_doc_matrix(self, vocab, documents):
        """Term counts with one row per vocabulary term and one column per document.

        Each document is a list of phrases (an intent with several example phrases,
        or a single question).
        """
        index = {term: i for i, term in enumerate(vocab)}
        matrix = np.zeros((len(vocab), len(documents)))
        for col, phrases in enumerate(documents):
            for phrase in phrases:
                for token in self.preprocess(phrase):
                    if token in index:
                        matrix[index[token], col] += 1
        return matrix

    def input_to_vector(self, user_input, vocab):
        """Term counts for the user's input, using the same vocabulary."""
        index = {term: i for i, term in enumerate(vocab)}
        vector = np.zeros(len(vocab))
        for token in self.preprocess(user_input):
            if token in index:
                vector[index[token]] += 1
        return vector

    def best_match(self, user_input, vocab, term_doc_matrix):
        """Index and cosine similarity of the document most similar to the input.

        Returns a score of 0 when the input shares no words with the vocabulary.
        """
        query = self.input_to_vector(user_input, vocab)
        query_norm = np.linalg.norm(query)
        doc_norms = np.linalg.norm(term_doc_matrix, axis=0)
        if query_norm == 0:
            return 0, 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            similarities = np.nan_to_num((query @ term_doc_matrix) / (query_norm * doc_norms))
        best = int(np.argmax(similarities))
        return best, float(similarities[best])
