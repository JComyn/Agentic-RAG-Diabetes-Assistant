from typing import List

_AMBIGUOUS_SINGLE_WORDS = {"effects", "effect", "it", "they", "this", "that", "impact", "impacts", "risks", "benefits"}


def split_into_subqueries(query: str) -> List[str]:
    """Naive query decomposition: split on common conjunctions and punctuation.

    Returns a list of subqueries (at least the original query if no split found).
    """
    if not query:
        return []

    q = query.strip()
    # Split on ' and ' (common conjunction) and commas
    parts = []
    for sep in [' and ', ',', ';', ' / '] :
        if sep in q.lower():
            # Respect original casing slightly by splitting on lower-cased version
            parts = [p.strip() for p in q.replace(' AND ', ' and ').split(sep) if p.strip()]
            break
    if not parts:
        # Fallback: return original query as single-element list
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
