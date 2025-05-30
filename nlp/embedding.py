import csv
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from nlp.utils import normalize

class EmbeddingSimilarity:
    def __init__(self, qna_path):
        self.questions = []
        self.answers = []
        self.qna_path = qna_path
        self.embeddings_path = qna_path.replace('.csv', '_embeddings.npy')
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_md")
        except IOError:
            raise RuntimeError("spaCy model 'en_core_web_md' not found. Run: python -m spacy download en_core_web_md")
        
        self._load_qna()
        self._load_or_generate_embeddings()

    def _load_qna(self):
        """Load questions and answers from CSV"""
        if not os.path.exists(self.qna_path):
            raise FileNotFoundError(f"Q/A file not found: {self.qna_path}")
        
        with open(self.qna_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.questions.append(row['question'])
                self.answers.append(row['answer'])

    def _load_or_generate_embeddings(self):
        """Load embeddings from .npy file or generate them"""
        if os.path.exists(self.embeddings_path):
            # Check if embeddings file is newer than qna.csv
            embeddings_mtime = os.path.getmtime(self.embeddings_path)
            qna_mtime = os.path.getmtime(self.qna_path)
            
            if embeddings_mtime >= qna_mtime:
                print(f"Loading cached embeddings from {self.embeddings_path}")
                self.question_embeddings = np.load(self.embeddings_path)
                return
        
        print(f"Generating embeddings for {len(self.questions)} questions...")
        self.question_embeddings = np.array([
            self.nlp(question).vector for question in self.questions
        ])
        
        # Save embeddings for future use
        np.save(self.embeddings_path, self.question_embeddings)
        print(f"Embeddings saved to {self.embeddings_path}")

    def _char_similarity(self, str1, str2):
        """Calculate character-level similarity between two strings"""
        str1 = str1.lower().replace("?", "").replace("'", "")
        str2 = str2.lower().replace("?", "").replace("'", "")
        
        # Simple character overlap similarity
        set1 = set(str1)
        set2 = set(str2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0

    def reply(self, user_input, threshold=0.6):
        """Find the best answer using embedding similarity"""
        user_input = normalize(user_input)
        if not user_input.strip():
            return None
            
        user_embedding = self.nlp(user_input).vector.reshape(1, -1)
        similarities = cosine_similarity(user_embedding, self.question_embeddings)[0]
        
        # Get top candidates (scores within 0.01 of the best)
        best_score = np.max(similarities)
        if best_score < threshold:
            return None
            
        # Find all candidates within 0.01 of the best score
        top_candidates = []
        for i, score in enumerate(similarities):
            if score >= best_score - 0.01:  # Within 0.01 of best
                top_candidates.append((i, score, self.questions[i], self.answers[i]))
        
        # If only one candidate, return it
        if len(top_candidates) == 1:
            return top_candidates[0][3]
        
        # Multiple candidates - use tie-breaking
        user_words = set(user_input.lower().split())
        best_candidate = None
        best_score_combined = -1
        
        for idx, score, question, answer in top_candidates:
            # Word overlap score
            question_words = set(question.lower().split())
            word_overlap = len(user_words.intersection(question_words))
            
            # Character similarity score for main content words
            user_content = user_input.lower().replace("what's", "").replace("what", "").replace("a", "").strip()
            question_content = question.lower().replace("what's", "").replace("what", "").replace("a", "").strip()
            char_sim = self._char_similarity(user_content, question_content)
            
            # Combined score: word overlap + character similarity
            combined_score = word_overlap + char_sim
            
            if combined_score > best_score_combined:
                best_score_combined = combined_score
                best_candidate = (idx, score, question, answer)
        
        if best_candidate:
            return best_candidate[3]  # Return the answer
        else:
            return self.answers[np.argmax(similarities)]

    def get_best_similarity_score(self, user_input):
        """Get the best similarity score for debugging/testing"""
        user_input = normalize(user_input)
        if not user_input.strip():
            return 0.0
            
        user_embedding = self.nlp(user_input).vector.reshape(1, -1)
        similarities = cosine_similarity(user_embedding, self.question_embeddings)[0]
        return np.max(similarities) 