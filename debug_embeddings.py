#!/usr/bin/env python3
"""
Debug embedding similarity to understand why wrong utensils are being matched
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nlp.embedding import EmbeddingSimilarity

def debug_embeddings():
    print("🔍 EMBEDDING SIMILARITY INVESTIGATION")
    print("=" * 55)
    
    # Load the embedding system
    embed_sim = EmbeddingSimilarity('qna.csv')
    
    # Test queries that are giving wrong answers
    test_queries = [
        "what flips pancakes?",      # Got: tongs, Should be: fish slice/spatula
        "what drains pasta?",        # Got: fish slice, Should be: colander  
        "What's a colendar",         # Got: blender, Should be: colander
        "tool for draining pasta",   # Should be: colander
        "utensil for flipping food"  # Should be: fish slice/spatula
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 50)
        
        # Get user embedding
        user_embedding = embed_sim.nlp(query).vector.reshape(1, -1)
        
        # Calculate similarities to all questions
        similarities = cosine_similarity(user_embedding, embed_sim.question_embeddings)[0]
        
        # Get top 5 matches
        top_indices = np.argsort(similarities)[::-1][:5]
        
        print("Top 5 most similar questions:")
        for i, idx in enumerate(top_indices, 1):
            score = similarities[idx]
            question = embed_sim.questions[idx]
            answer = embed_sim.answers[idx]
            print(f"{i}. Score: {score:.3f}")
            print(f"   Q: {question}")
            print(f"   A: {answer}")
            print()
        
        # Show what the system would return
        result = embed_sim.reply(query, threshold=0.6)
        print(f"🤖 System returns: {result}")
        print(f"🎯 Expected: Should mention the correct utensil")

if __name__ == "__main__":
    debug_embeddings() 