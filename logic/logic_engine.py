"""
Logic engine for fact assertion, checking, and fuzzy safety queries.
"""
from typing import Optional
import os
from nltk.sem import Expression
from nltk.inference import ResolutionProver
from .aliases import canonical_name
from .fuzzy_safety import safety_score # Import the fuzzy safety score function

def strip_articles(text: str) -> str:
    # Remove leading articles from a string
    articles = ('a ', 'an ', 'the ')
    for article in articles:
        if text.startswith(article):
            return text[len(article):]
    return text

def _to_camel_case_prop(prop_phrase: str) -> str:
    """Converts a property phrase like 'microwave safe' to 'MicrowaveSafe'."""
    return "".join(p.capitalize() for p in prop_phrase.split())

def assert_fact(text: str) -> str:
    """Assert a new fact into the knowledge base, checking for contradictions."""
    text = text.strip().lower().replace('?', '')
    if text.startswith('i know that '):
        rest = text[len('i know that '):]
        
        fol_str = None
        prop_is_negated = False

        if ' is not ' in rest:
            parts = rest.split(' is not ', 1)
            prop_is_negated = True
        elif ' are not ' in rest:
            parts = rest.split(' are not ', 1)
            prop_is_negated = True
        elif ' is ' in rest:
            parts = rest.split(' is ', 1)
        elif ' are ' in rest:
            parts = rest.split(' are ', 1)
        else:
            # Fallback for "I know that X Y" (e.g. "I know that tongs metal")
            parts = rest.split(None, 1) 
            if len(parts) != 2:
                return "Sorry, I can only remember facts in the form 'I know that X [is/are/is not/are not] Y' or 'I know that X Y'."

        if len(parts) == 2:
            utensil = canonical_name(strip_articles(parts[0].strip()))
            prop = _to_camel_case_prop(parts[1].strip())
            fol_str = f"{prop}({utensil})"
            if prop_is_negated:
                fol_str = f"~{fol_str}"
        else:
            return "Sorry, I can only remember facts in the form 'I know that X [is/are/is not/are not] Y'."

        try:
            kb = get_kb()
            expr = read_expr(fol_str)
            # For contradiction, check the opposite form
            neg_expr_str = f"{prop}({utensil})" if prop_is_negated else f"~{prop}({utensil})"
            neg_expr = read_expr(neg_expr_str)
            
            if kb.entails(neg_expr):
                return f"Sorry, that contradicts what I already know."
            if expr not in kb.facts: # Avoid adding duplicates to in-memory KB
                kb.facts.append(expr)
            return f"OK, I'll remember that {parts[0].strip()} {parts[1].strip()}{ ' not' if prop_is_negated else ''}."
        except Exception as e:
            # print(f"Error in assert_fact: {e}") # Optional debug
            return "Sorry, I couldn't process that fact."
            
    return "Sorry, I can only remember facts in the form 'I know that X [is/are/is not/are not] Y' or 'I know that X Y'."

def check_fact(text: str) -> str:
    """Check a fact against the knowledge base. Return Correct/Incorrect/Unknown."""
    kb = get_kb()
    fol_str = parse_fact_from_text(text)
    if not fol_str:
        return "Unknown."
    try:
        query = read_expr(fol_str)
    except Exception:
        return "Unknown."
    print("Parsed FOL string:", fol_str)
    print("Query:", query)
    neg_query = read_expr("~" + str(query))
    print("Negated Query:", neg_query)
    if kb.entails(query):
        return "Correct."
    if kb.entails(neg_query):
        return "Incorrect."
    return "Unknown."

# Replace the existing stub `safety_query` with the new helper and handler.
# Helper function to get sharpness and grip from FOL KB
def _get_sharpness_grip_values(utensil_name: str) -> tuple[float, float]:
    """
    Queries the FOL KB for sharpness and grip properties of a utensil.
    Returns numerical values (sharpness, grip).
    Defaults: sharpness=5 (medium), grip=5 (average).
    """
    kb = get_kb()
    
    # Default values
    sharpness_val = 5.0 # Medium sharpness by default
    grip_val = 5.0      # Average grip

    # Check for Sharp(utensil)
    sharp_query_str = f"Sharp({utensil_name})"
    try:
        if kb.entails(read_expr(sharp_query_str)):
            sharpness_val = 9.0 # Very sharp
    except Exception:
        pass # Keep default if parsing fails or fact not found

    # Check for NotSharp(utensil) - this is ~Sharp(utensil)
    not_sharp_query_str = f"~Sharp({utensil_name})"
    try:
        if kb.entails(read_expr(not_sharp_query_str)):
            sharpness_val = 1.0 # Explicitly not sharp
    except Exception:
        pass

    # Check for GoodGrip(utensil)
    good_grip_query_str = f"GoodGrip({utensil_name})"
    try:
        if kb.entails(read_expr(good_grip_query_str)):
            grip_val = 8.0 # Good grip
    except Exception:
        pass

    # Check for PoorGrip(utensil)
    poor_grip_query_str = f"PoorGrip({utensil_name})"
    try:
        if kb.entails(read_expr(poor_grip_query_str)):
            grip_val = 3.0 # Poor grip
    except Exception:
        pass
        
    return sharpness_val, grip_val

def get_fuzzy_safety_reply(user_input: str) -> Optional[str]:
    """
    Parses a fuzzy safety query, gets sharpness/grip from KB,
    calls the fuzzy logic system, and formats the reply.
    Example query: "Is kitchenknife safe for children?" 
                   or "How safe is a kitchenknife?"
    """
    text = user_input.strip().lower().replace('?', '')
    
    utensil_name = None
    patterns = ["is ", "how safe is "]
    suffix_triggers = [" safe for children", " safe to use", " safe"]

    parsed_utensil = False
    for pat_start in patterns:
        if text.startswith(pat_start):
            potential_utensil_phrase = text[len(pat_start):]
            for pat_end in suffix_triggers:
                if potential_utensil_phrase.endswith(pat_end):
                    utensil_name = canonical_name(strip_articles(potential_utensil_phrase[:-len(pat_end)].strip()))
                    parsed_utensil = True
                    break
            if parsed_utensil:
                break
            if pat_start == "how safe is ": # Handles "how safe is [utensil]"
                 utensil_name = canonical_name(strip_articles(potential_utensil_phrase.strip()))
                 parsed_utensil = True
                 break

    if not utensil_name:
        # Fallback parsing for simpler queries like "is kitchen knife safe"
        words = text.split()
        if len(words) > 1:
            if words[-1] == "safe" and words[0] == "is":
                potential_utensil = strip_articles(" ".join(words[1:-1]))
                utensil_name = canonical_name(potential_utensil)
            elif len(words) > 2 and words[-2] == "safe" and words[0] == "is": # e.g. "is kitchen knife safe to use"
                potential_utensil = strip_articles(" ".join(words[1:-2]))
                utensil_name = canonical_name(potential_utensil)

    if not utensil_name:
        return None 

    sharpness, grip = _get_sharpness_grip_values(utensil_name)
    
    safety_label, safety_val = safety_score(sharpness, grip)
    return f"{utensil_name.capitalize()} safety rating: {safety_label} (score: {safety_val:.2f})"

def reply(user_text: str) -> Optional[str]:
    """Main entry point for logic engine. Returns a response or None."""
    # TODO: Implement logic routing
    return None 

read_expr = Expression.fromstring

class FOLKnowledgeBase:
    def __init__(self, kb_path):
        self.kb_path = kb_path
        self.facts = []
        self._load_kb()

    def _load_kb(self):
        self.facts = []
        if not os.path.exists(self.kb_path):
            return
        with open(self.kb_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    expr = read_expr(line)
                    self.facts.append(expr)
                except Exception:
                    continue  # skip lines that can't be parsed
        print(f"[DEBUG] Loaded {len(self.facts)} FOL facts from {self.kb_path}")
        for fact in self.facts:
            print(f"[DEBUG] KB fact: {fact}")

    def entails(self, query):
        # Returns True if KB entails the query
        return ResolutionProver().prove(query, self.facts)

# Singleton KB instance
_kb = None

def get_kb():
    global _kb
    if _kb is None:
        kb_path = os.path.join(os.path.dirname(__file__), 'logical-kb.csv')
        _kb = FOLKnowledgeBase(kb_path)
    return _kb

def parse_fact_from_text(text: str):
    text = text.strip().lower().replace('?', '')

    if text.startswith('check that '):
        rest = text[len('check that '):]
        
        # Order of checks: specific (with "not") before general
        if ' is not ' in rest:
            parts = rest.split(' is not ', 1)
            if len(parts) == 2:
                utensil = canonical_name(strip_articles(parts[0].strip()))
                prop = _to_camel_case_prop(parts[1].strip())
                return f"~{prop}({utensil})"
        elif ' are not ' in rest:
            parts = rest.split(' are not ', 1)
            if len(parts) == 2:
                utensil = canonical_name(strip_articles(parts[0].strip()))
                prop = _to_camel_case_prop(parts[1].strip())
                return f"~{prop}({utensil})"
        elif ' is ' in rest:
            parts = rest.split(' is ', 1)
            if len(parts) == 2:
                utensil = canonical_name(strip_articles(parts[0].strip()))
                prop = _to_camel_case_prop(parts[1].strip())
                return f"{prop}({utensil})"
        elif ' are ' in rest:
            parts = rest.split(' are ', 1)
            if len(parts) == 2:
                utensil = canonical_name(strip_articles(parts[0].strip()))
                prop = _to_camel_case_prop(parts[1].strip())
                return f"{prop}({utensil})"
        else: # Fallback for "check that X Y" (e.g. "check that tongs metal")
            parts = rest.split(None, 1) # Split on first whitespace sequence
            if len(parts) == 2: # Expecting Utensil Property
                utensil = canonical_name(strip_articles(parts[0].strip()))
                prop = _to_camel_case_prop(parts[1].strip())
                return f"{prop}({utensil})"

    # Keep existing logic for "is X Y" or "is a X Y" if needed, or refactor similarly
    # For now, focusing on "check that"
    # Example refactor for "is X Y":
    if text.startswith('is ') or text.startswith('are '):
        copula_len = 3 if text.startswith('is ') else 4
        rest_of_query = text[copula_len:]
        # This part needs careful thought on how to split Utensil from Property
        # Assuming Utensil is the first word token for simplicity here
        parts = rest_of_query.split(None, 1)
        if len(parts) == 2:
            utensil = canonical_name(strip_articles(parts[0].strip()))
            prop = _to_camel_case_prop(parts[1].strip())
            return f"{prop}({utensil})"
            
    return None 