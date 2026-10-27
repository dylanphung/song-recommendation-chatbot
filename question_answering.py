"""Answers questions by finding the most similar question in a question-answer dataset."""
import pandas as pd

from utils import Utils

THRESHOLD = 0.1  # minimum cosine similarity to accept a match


class QuestionAnsweringSystem:
    def __init__(self, dataset_filename, fallback="I'm not sure about that. Could you rephrase the question?"):
        data = pd.read_csv(dataset_filename)
        self.questions = data["Question"].astype(str).tolist()
        self.answers = data["Answer"].astype(str).tolist()
        self.fallback = fallback

        self.utils = Utils()
        self.vocab = self.utils.build_vocabulary(self.questions)
        # one column per question
        self.term_doc_matrix = self.utils.build_term_doc_matrix(self.vocab, [[q] for q in self.questions])

    def answer_question(self, user_question):
        best, score = self.utils.best_match(user_question, self.vocab, self.term_doc_matrix)
        return self.answers[best] if score > THRESHOLD else self.fallback
