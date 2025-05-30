import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

TFIDF_THRESHOLD = 0.65  # Centralized threshold for all TF-IDF similarity checks

class TfidfSimilarity:
    def __init__(self, qna_path):
        self.questions = []
        self.answers = []
        self.vectorizer = TfidfVectorizer()
        self.qna_path = qna_path
        self._fit()

    def _fit(self):
        if not os.path.exists(self.qna_path):
            raise FileNotFoundError(f"Q/A file not found: {self.qna_path}")
        with open(self.qna_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.questions.append(row['question'])
                self.answers.append(row['answer'])
        self.question_vecs = self.vectorizer.fit_transform(self.questions)

    def reply(self, user_input, threshold=TFIDF_THRESHOLD):
        user_vec = self.vectorizer.transform([user_input])
        sims = cosine_similarity(user_vec, self.question_vecs)[0]
        best_idx = sims.argmax()
        if sims[best_idx] >= threshold:
            return self.answers[best_idx]
        return None

    def get_best_similarity_score(self, user_input):
        """Get the best similarity score for debugging/testing"""
        if not user_input.strip():
            return 0.0
        user_vec = self.vectorizer.transform([user_input])
        sims = cosine_similarity(user_vec, self.question_vecs)[0]
        return sims.max() 