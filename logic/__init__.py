"""
Logic reasoning module for the Kitchen Utensils Chatbot.
Exposes main reply() and utility functions.
"""
from .logic_engine import reply, assert_fact, check_fact, get_fuzzy_safety_reply
from .fuzzy_safety import safety_score
from .aliases import canonical_name 