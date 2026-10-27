"""Works out what the user wants by comparing their input with example phrases for each intent."""
from utils import Utils

# Top-level intents, with example phrases for each
MAIN_INTENTS = {
    "song_recommendation": ["recommend"],
    "rec_list": ["clear", "remove", "delete", "show", "display", "view", "empty"],
    "question": ["What is", "How do", "What are", "Tell me about", "Explain", "Where can I find",
                 "Why does", "Who is", "When was", "How many", "How long"],
    "small_talk": ["Hi", "Hello", "How are you", "Tell me a joke", "What's up", "Good morning",
                   "What can you do", "What's my name"],
}

# Which kind of recommendation the user asked for
RECOMMENDATION_INTENTS = {
    "genre": ["by genre"],
    "artist": ["by artist"],
    "popular": ["popular"],
    "similar": ["similar"],
}

# What the user wants to do with their recommendation list
LIST_INTENTS = {
    "display": ["show", "display", "view"],
    "clear": ["clear", "empty"],
    "delete": ["delete", "remove"],
}

THRESHOLD = 0.1  # minimum cosine similarity to accept a match


class IntentMatching:
    def __init__(self):
        self.utils = Utils()
        self.main = self._build_model(MAIN_INTENTS)
        self.recommendation = self._build_model(RECOMMENDATION_INTENTS)
        self.list = self._build_model(LIST_INTENTS)

    def _build_model(self, intents):
        """Vocabulary and term-document matrix (one column per intent) for a set of intents."""
        phrases = [phrase for examples in intents.values() for phrase in examples]
        vocab = self.utils.build_vocabulary(phrases)
        matrix = self.utils.build_term_doc_matrix(vocab, list(intents.values()))
        return list(intents), vocab, matrix

    def _classify(self, user_input, model):
        names, vocab, matrix = model
        best, score = self.utils.best_match(user_input, vocab, matrix)
        return names[best] if score >= THRESHOLD else "unknown"

    def get_intent(self, user_input):
        return self._classify(user_input, self.main)

    def get_rec_intent(self, user_input):
        return self._classify(user_input, self.recommendation)

    def get_list_intent(self, user_input):
        return self._classify(user_input, self.list)
