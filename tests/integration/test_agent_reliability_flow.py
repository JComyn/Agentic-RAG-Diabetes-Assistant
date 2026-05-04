import sys
import types
import pytest

# evaluation.py only imports standard library at module level, so the import is safe
# without any stubs in place.
from evaluation import evaluation


class _DummyRunnableConfig:
    def __init__(self, recursion_limit=None):
        self.recursion_limit = recursion_limit


@pytest.fixture(autouse=True)
def _reset_rag_cache(monkeypatch):
    """Reset the RAG runtime cache before each test so stubs set via monkeypatch take effect."""
    monkeypatch.setattr(evaluation, '_RAG_RUNTIME_CACHE', None)


def test_low_evidence_returns_conservative_language(monkeypatch):
    """When the system has no retrieved contexts, the agent should respond with conservative language."""
    # Build a stub app whose invoke returns a confident answer but no validated_docs
    class StubApp:
        def invoke(self, inputs, config=None):
            return {"final_answer": "You should start insulin immediately.", "validated_docs": []}

    # Patch load_rag_runtime_dependencies so run_system_for_evaluation uses our stub app
    monkeypatch.setattr(
        evaluation,
        'load_rag_runtime_dependencies',
        lambda: (StubApp(), 10, _DummyRunnableConfig),
    )

    answer, contexts = evaluation.run_system_for_evaluation("What treatment should I start?")

    # Expect no contexts and conservative language in the answer
    assert contexts == []
    conservative_phrases = ["don't have enough", "not enough", "i am not sure", "cannot answer", "i don't know", "not confident"]
    assert any(p in answer.lower() for p in conservative_phrases), f"Answer was not conservative: {answer}"
