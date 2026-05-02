from typing import List, Dict


def merge_dedup_and_score(documents: List[object], top_k: int) -> List[object]:
    """Merge documents with identical content and compute evidence scores.

    Behavior:
    - Documents with identical page_content (after stripping) are merged into one.
    - Scores are taken from metadata['score'] when present (numeric), default 0.
    - Merged score is the sum of individual scores (simple evidence accumulation).
    - Metadata sources are combined into a list under metadata['sources'].
    - Returns top_k documents sorted by descending score.
    """
    if not documents:
        return []

    from types import SimpleNamespace
    merged: Dict[str, object] = {}

    for doc in documents:
        key = (doc.page_content or "").strip()
        score = 0.0
        try:
            score = float(doc.metadata.get("score", 0)) if doc.metadata else 0.0
        except Exception:
            score = 0.0

        source = None
        try:
            source = doc.metadata.get("source") if doc.metadata else None
        except Exception:
            source = None

        if key in merged:
            existing = merged[key]
            existing_score = float(existing.metadata.get("score", 0)) if existing.metadata else 0.0
            new_score = existing_score + score
            # Update score
            existing.metadata["score"] = new_score
            # Merge sources
            existing_sources = existing.metadata.get("sources") or ([] if existing.metadata.get("source") is not None else [])
            # If original had single 'source' field, include it
            if existing.metadata.get("source") is not None and not existing_sources:
                existing_sources = [existing.metadata.get("source")]
            if source is not None:
                existing_sources.append(source)
            existing.metadata["sources"] = existing_sources
            # remove single source key to avoid confusion
            if "source" in existing.metadata:
                existing.metadata.pop("source")
        else:
            # Create a shallow copy of metadata to avoid mutating original
            meta = dict(doc.metadata) if doc.metadata else {}
            meta["score"] = float(meta.get("score", 0))
            # Keep source in sources list for consistency
            src = meta.get("source")
            if src is not None:
                meta["sources"] = [src]
                meta.pop("source", None)
            merged_doc = SimpleNamespace(page_content=doc.page_content, metadata=meta)
            merged[key] = merged_doc

    # Convert to list and sort by score descending
    merged_list = list(merged.values())
    merged_list.sort(key=lambda d: float(d.metadata.get("score", 0)), reverse=True)

    # Apply top_k
    if top_k is not None and top_k > 0:
        merged_list = merged_list[:top_k]

    return merged_list
