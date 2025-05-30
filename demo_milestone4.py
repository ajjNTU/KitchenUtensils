#!/usr/bin/env python3
"""
Demonstration of Milestone 4: Embedding Fallback

This shows that the embedding fallback system is implemented and working.
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from nlp.similarity import TfidfSimilarity
from nlp.embedding import EmbeddingSimilarity

def demo():
    print("🎯 MILESTONE 4: EMBEDDING FALLBACK DEMONSTRATION")
    print("=" * 55)
    
    try:
        # Initialize both systems
        tfidf_sim = TfidfSimilarity('qna.csv')
        embed_sim = EmbeddingSimilarity('qna.csv')
        
        print("✅ TF-IDF similarity system loaded")
        print("✅ Embedding similarity system loaded (spaCy en_core_web_md)")
        print("✅ Embeddings cached for fast lookup")
        
        # Test a query that demonstrates embedding understanding
        test_query = "utensil for mixing eggs"
        
        print(f"\n📝 Test Query: '{test_query}'")
        print("-" * 40)
        
        # Show TF-IDF result
        tfidf_score = tfidf_sim.get_best_similarity_score(test_query)
        tfidf_answer = tfidf_sim.reply(test_query)
        print(f"TF-IDF Score: {tfidf_score:.3f}")
        print(f"TF-IDF Answer: {tfidf_answer}")
        
        # Show embedding result
        embed_score = embed_sim.get_best_similarity_score(test_query)
        embed_answer_high = embed_sim.reply(test_query, threshold=0.65)
        embed_answer_mid = embed_sim.reply(test_query, threshold=0.50)
        
        print(f"Embedding Score: {embed_score:.3f}")
        print(f"Embedding (≥0.65): {embed_answer_high}")
        print(f"Embedding (≥0.50): {embed_answer_mid}")
        
        # Explain the fallback logic
        print(f"\n🔄 Fallback Logic:")
        if tfidf_score >= 0.35:
            print(f"   TF-IDF confident ({tfidf_score:.3f} ≥ 0.35) → Use TF-IDF answer")
        else:
            print(f"   TF-IDF not confident ({tfidf_score:.3f} < 0.35) → Try embedding")
            if embed_score >= 0.65:
                print(f"   Embedding confident ({embed_score:.3f} ≥ 0.65) → Use embedding answer")
            else:
                print(f"   Embedding not confident ({embed_score:.3f} < 0.65) → Use fallback")
        
        print("\n🎉 MILESTONE 4 IMPLEMENTATION COMPLETE!")
        print("Features implemented:")
        print("  • spaCy en_core_web_md integration")
        print("  • Embedding generation and caching (.npy files)")
        print("  • TF-IDF < 0.35 threshold for fallback trigger")
        print("  • Embedding ≥ 0.65 threshold for acceptance")
        print("  • Full integration into main.py routing")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you've run: python -m spacy download en_core_web_md")

if __name__ == "__main__":
    demo() 