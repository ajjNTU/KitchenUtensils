"""
Fuzzy safety model for kitchen utensils using Simpful.
"""
from simpful import *

# Create a fuzzy system
FS = FuzzySystem()

# Input variables
FS.add_linguistic_variable("Sharpness", LinguisticVariable([
    TrapezoidFuzzySet(1, 1, 3, 4, term="low"),
    TriangleFuzzySet(3, 5, 7, term="med"),
    TrapezoidFuzzySet(6, 7, 10, 10, term="high")
], universe_of_discourse=[0, 10]))

FS.add_linguistic_variable("Grip", LinguisticVariable([
    TrapezoidFuzzySet(1, 1, 3, 5, term="poor"),
    TrapezoidFuzzySet(4, 6, 10, 10, term="good")
], universe_of_discourse=[0, 10]))

# Output variable
FS.add_linguistic_variable("Safety", LinguisticVariable([
    TrapezoidFuzzySet(0, 0, 0.3, 0.4, term="low"),
    TriangleFuzzySet(0.3, 0.5, 0.7, term="moderate"),
    TrapezoidFuzzySet(0.6, 0.7, 1, 1, term="high")
], universe_of_discourse=[0, 1]))

# Fuzzy rules
FS.add_rules([
    "IF (Sharpness IS high) AND (Grip IS poor) THEN (Safety IS low)",
    "IF (Sharpness IS high) AND (Grip IS good) THEN (Safety IS moderate)",
    "IF (Sharpness IS low) AND (Grip IS good) THEN (Safety IS high)",
    "IF (Sharpness IS low) AND (Grip IS poor) THEN (Safety IS moderate)",
    "IF (Sharpness IS med) AND (Grip IS good) THEN (Safety IS moderate)",
    "IF (Sharpness IS med) AND (Grip IS poor) THEN (Safety IS low)"
])

def safety_score(sharpness: float, grip: float) -> tuple[str, float]:
    """Return (safety_label, score) for given sharpness and grip values.
    Tries to use fuzzy memberships, falls back to crisp value thresholds if needed.
    """
    FS.set_variable("Sharpness", sharpness)
    FS.set_variable("Grip", grip)
    result = FS.inference()["Safety"]
    try:
        # Try to get the membership degrees for each label
        safety_lv = FS._variables["Safety"]  # Not public, but works for now
        memberships = safety_lv.fuzzify(result)
        best_label = max(memberships, key=memberships.get)
    except Exception:
        # Fallback: use crisp value and thresholds
        if result < 0.3:
            best_label = "low"
        elif result < 0.7:
            best_label = "moderate"
        else:
            best_label = "high"
    return (best_label, result) 