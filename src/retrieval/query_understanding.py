from typing import List
import re

_AMBIGUOUS_SINGLE_WORDS = {"effects", "effect", "it", "they", "this", "that", "impact", "impacts", "risks", "benefits"}


def split_into_subqueries(query: str) -> List[str]:
    """Naive query decomposition: split on common conjunctions and punctuation.

    Returns a list of subqueries (at least the original query if no split found).
    Uses case-insensitive splitting for conjunctions like 'and'.
    """
    if not query:
        return []

    q = query.strip()
    # Split on 'and' as a word (case-insensitive), commas, semicolons, or slashes
    # Use regex with IGNORECASE to handle 'And', 'AND', etc.
    pattern = r"\band\b|,|;|/"
    parts = [p.strip() for p in re.split(pattern, q, flags=re.IGNORECASE) if p.strip()]
    if not parts:
        return [q]
    return parts


def detect_ambiguity(query: str) -> bool:
    """Detects if a query is likely ambiguous.

    Heuristic: if the query is a single short word in a known ambiguous set, return True.
    """
    if not query:
        return False
    q = query.strip().lower()
    # single-word ambiguity
    if " " not in q and q in _AMBIGUOUS_SINGLE_WORDS:
        return True
    return False
