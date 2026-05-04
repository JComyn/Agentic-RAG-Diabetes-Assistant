from typing import List, Union

from src.config import JUDGE_UNSUPPORTED_CLAIM_THRESHOLD

# Simple Spanish stopwords set for lightweight text comparison
SPANISH_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "con", "y", "o", "de", "del",
    "en", "por", "para", "que", "se", "no", "a", "es", "los", "las",
    "su", "sus", "al", "lo", "como", "más", "menos"
}


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    words = [w.strip('.,;:()[]"\'"').lower() for w in text.split()]
    return [w for w in words if w and w not in SPANISH_STOPWORDS]


def classify_reliability(answer: str, evidence: Union[List[str], List[object]], threshold: float = JUDGE_UNSUPPORTED_CLAIM_THRESHOLD) -> str:
    """Classify the reliability of an answer given supporting evidence.

    Returns one of:
      - 'supported' : answer appears to be supported by the provided evidence
      - 'unsupported_claim' : answer contains claims not found in the evidence
      - 'insufficient_evidence' : evidence is empty or too small to judge

    This is a lightweight heuristic implementation used for testing and simple
    safety checks. It checks token overlap (ignoring common Spanish stopwords)
    between the answer and the evidence corpus.
    """
    # Normalize evidence to list of strings
    texts: List[str] = []
    for e in evidence or []:
        if hasattr(e, "page_content"):
            texts.append(getattr(e, "page_content") or "")
        else:
            texts.append(str(e or ""))

    # If there's no evidence, we can't support claims
    if not any(texts):
        return "insufficient_evidence"

    answer_tokens = set(_tokenize(answer))
    evidence_tokens = set()
    for t in texts:
        evidence_tokens.update(_tokenize(t))

    if not answer_tokens:
        # No meaningful content in answer
        return "insufficient_evidence"

    # Compute overlap ratio
    overlap = answer_tokens.intersection(evidence_tokens)
    overlap_ratio = len(overlap) / len(answer_tokens)

    # Heuristic thresholds: if less than the configured fraction of answer tokens appear in
    # evidence, treat as unsupported claim.
    if overlap_ratio < threshold:
        return "unsupported_claim"

    return "supported"
