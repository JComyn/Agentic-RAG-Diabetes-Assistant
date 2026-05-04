import pytest
from src.retrieval.pipeline import merge_dedup_and_score


class Document:
    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = dict(metadata) if metadata else {}


def test_merge_dedup_and_score_combines_duplicates_and_sums_scores():
    # Two documents with identical content but different source/score
    content = "This is a test document about diabetes."
    doc1 = Document(page_content=content, metadata={"source": "local_a", "score": 0.6})
    doc2 = Document(page_content=content, metadata={"source": "local_b", "score": 0.4})
    # Another unique document
    doc3 = Document(page_content="Different content", metadata={"source": "local_c", "score": 0.5})

    merged = merge_dedup_and_score([doc1, doc2, doc3], top_k=10)

    # Expect duplicates merged => 2 documents
    assert len(merged) == 2

    # Find merged doc by content
    merged_doc = next(d for d in merged if d.page_content == content)
    # Expect the scores to be summed
    assert float(merged_doc.metadata.get("score", 0)) == pytest.approx(1.0)

    # Check other doc remains with its original score
    other_doc = next(d for d in merged if d.page_content == "Different content")
    assert float(other_doc.metadata.get("score", 0)) == pytest.approx(0.5)
