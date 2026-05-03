import sys
import types
import pytest

# Inject minimal stub modules to allow importing evaluation.evaluation in environments
# without the full set of heavy dependencies installed.
stub_names = [
    'datasets', 'ragas', 'ragas.metrics', 'langchain_core', 'langchain_core.runnables',
    'langchain_openai', 'langchain_openai.embeddings', 'src', 'src.graph', 'src.config'
]
for name in stub_names:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

# Provide minimal attributes the evaluation module expects
sys.modules['datasets'].Dataset = lambda *a, **k: None
# ragas.evaluate should return an object with to_pandas method; provide a simple stub later if used
rain = types.ModuleType('ragas')
setattr(rain, 'evaluate', lambda *a, **k: types.SimpleNamespace(to_pandas=lambda: None))
sys.modules['ragas'] = rain
# ragas.metrics placeholders
metrics = types.ModuleType('ragas.metrics')
for attr in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'ContextRelevance', 'answer_correctness', 'answer_similarity']:
    setattr(metrics, attr, types.SimpleNamespace(name=attr))
sys.modules['ragas.metrics'] = metrics

# Provide a minimal RunnableConfig and config constants
class DummyRunnableConfig:
    def __init__(self, recursion_limit=None):
        self.recursion_limit = recursion_limit

sys.modules['langchain_core.runnables'].RunnableConfig = DummyRunnableConfig

# Provide minimal src.config and src.graph.app
src_mod = types.ModuleType('src')
config_mod = types.ModuleType('src.config')
setattr(config_mod, 'RECURSION_LIMIT', 10)
sys.modules['src.config'] = config_mod

graph_mod = types.ModuleType('src.graph')
# app.invoke will be monkeypatched in the test
class DummyApp:
    def invoke(self, inputs, config=None):
        return {'final_answer': 'stub', 'validated_docs': []}

setattr(graph_mod, 'app', DummyApp())
sys.modules['src.graph'] = graph_mod

# Now import the evaluation module
from evaluation import evaluation


def test_low_evidence_returns_conservative_language(monkeypatch):
    """When the system has no retrieved contexts, the agent should respond with conservative language."""
    # Monkeypatch the app.invoke to return a confident answer but no validated_docs
    monkeypatch.setattr(evaluation.app, 'invoke', lambda inputs, config=None: {"final_answer": "You should start insulin immediately.", "validated_docs": []})

    answer, contexts = evaluation.run_system_for_evaluation("What treatment should I start?")

    # Expect no contexts and conservative language in the answer
    assert contexts == []
    conservative_phrases = ["don't have enough", "not enough", "i am not sure", "cannot answer", "i don't know", "not confident"]
    assert any(p in answer.lower() for p in conservative_phrases), f"Answer was not conservative: {answer}"
