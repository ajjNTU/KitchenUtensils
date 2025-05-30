import re
from spellchecker import SpellChecker

spell = SpellChecker()

CONTRACTIONS = {
    "whats": "what is",
    "cant": "cannot",
    "wont": "will not",
    "isnt": "is not",
    "arent": "are not",
    "doesnt": "does not",
    "dont": "do not",
    "didnt": "did not",
    "hasnt": "has not",
    "havent": "have not",
    "hadnt": "had not",
    "shouldnt": "should not",
    "wouldnt": "would not",
    "couldnt": "could not",
    "mustnt": "must not",
    "wasnt": "was not",
    "werent": "were not",
    "lets": "let us",
    "im": "i am",
    "youre": "you are",
    "theyre": "they are",
    "weve": "we have",
    "ive": "i have",
    "youve": "you have",
    "theyve": "they have",
    "ill": "i will",
    "youll": "you will",
    "theyll": "they will",
    "shes": "she is",
    "hes": "he is",
    "its": "it is",
    "thats": "that is",
    "theres": "there is",
    "heres": "here is"
}

def expand_contractions(text):
    for contraction, expanded in CONTRACTIONS.items():
        text = re.sub(rf"\b{contraction}\b", expanded, text)
    return text

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = expand_contractions(text)
    # Spellcheck each word
    corrected = " ".join([spell.correction(word) or word for word in text.split()])
    return corrected 