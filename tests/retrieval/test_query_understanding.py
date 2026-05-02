import pytest

from src.retrieval.query_understanding import split_into_subqueries, detect_ambiguity


def test_split_into_subqueries_compound_query():
    query = "What are the symptoms and treatments for diabetes?"
    parts = split_into_subqueries(query)
    assert isinstance(parts, list)
    assert len(parts) > 1


def test_detect_ambiguity_single_word():
    assert detect_ambiguity("effects") is True


def test_split_mixed_case_and():
    query = "Causes And Treatments for asthma"
    parts = split_into_subqueries(query)
    assert isinstance(parts, list)
    assert len(parts) == 2
    assert parts[0].lower().startswith("causes")
    assert "treatments" in parts[1].lower()


def test_detect_ambiguity_non_ambiguous():
    # multi-word query shouldn't be considered ambiguous even if it contains an ambiguous token
    assert detect_ambiguity("diabetes symptoms") is False


def test_integration_split_and_ambiguity():
    query = "effects and treatments"
    parts = split_into_subqueries(query)
    ambiguous = detect_ambiguity(query)
    assert isinstance(parts, list)
    assert isinstance(ambiguous, bool)
    assert len(parts) > 1
    # combined query is not single-word ambiguous
    assert ambiguous is False


def test_transform_query_node_integration(monkeypatch):
    import sys
    import types

    # Create a fake components module to avoid initializing real LLMs or network calls
    fake_components = types.ModuleType("src.components")
    fake_components.llm = object()
    fake_components.tavily_client = None
    fake_components.final_retriever = None
    fake_components.USE_RERANKER = False
    sys.modules["src.components"] = fake_components

    # Fake chain returned by prompt | llm | parser
    class FakeChain:
        def __or__(self, other):
            return self

        def invoke(self, inputs, config=None):
            # Deterministic transformed query text
            return "TRANSFORMED QUERY TEXT"

    # Prepare a minimal fake langchain_core package with required symbols so importing src.agent doesn't try to load real libraries
    langcore = types.ModuleType("langchain_core")
    prompts = types.ModuleType("langchain_core.prompts")
    messages = types.ModuleType("langchain_core.messages")
    output_parsers = types.ModuleType("langchain_core.output_parsers")
    documents = types.ModuleType("langchain_core.documents")
    runnables = types.ModuleType("langchain_core.runnables")

    # Minimal message classes
    class BaseMessage:
        pass

    class SystemMessage:
        def __init__(self, content=None):
            self.content = content

    class HumanMessage:
        def __init__(self, content=None):
            self.content = content

    messages.BaseMessage = BaseMessage
    messages.SystemMessage = SystemMessage
    messages.HumanMessage = HumanMessage

    # Minimal output parsers
    class StrOutputParser:
        pass

    class JsonOutputParser:
        def with_fallbacks(self, f):
            return self

    output_parsers.StrOutputParser = StrOutputParser
    output_parsers.JsonOutputParser = JsonOutputParser

    # Minimal Document
    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    documents.Document = Document

    # Minimal runnables
    class RunnableConfig:
        pass

    class RunnableLambda:
        def __init__(self, fn):
            self.fn = fn

        def __or__(self, other):
            return self

        def invoke(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

    runnables.RunnableConfig = RunnableConfig
    runnables.RunnableLambda = RunnableLambda

    # Chat prompt template that returns a FakeChain when piped
    class MessagesPlaceholder:
        def __init__(self, variable_name=None):
            self.variable_name = variable_name

    prompts.MessagesPlaceholder = MessagesPlaceholder

    class FakePrompt:
        def __or__(self, other):
            return FakeChain()

    prompts.ChatPromptTemplate = type("ChatPromptTemplate", (), {"from_messages": classmethod(lambda cls, msgs: FakePrompt())})

    # Register fake langchain_core submodules in sys.modules
    sys.modules["langchain_core"] = langcore
    sys.modules["langchain_core.prompts"] = prompts
    sys.modules["langchain_core.messages"] = messages
    sys.modules["langchain_core.output_parsers"] = output_parsers
    sys.modules["langchain_core.documents"] = documents
    sys.modules["langchain_core.runnables"] = runnables

    # Also fake langgraph.graph to satisfy imports in src.agent
    langgraph = types.ModuleType("langgraph")
    langgraph_graph = types.ModuleType("langgraph.graph")
    langgraph_graph.END = object()
    sys.modules["langgraph"] = langgraph
    sys.modules["langgraph.graph"] = langgraph_graph

    # Import the function under test after fakes are in sys.modules
    from src.agent import transform_query_node

    state = {"original_query": "effects and treatments", "chat_history": []}
    result = transform_query_node(state, config=None)

    assert isinstance(result, dict)
    assert 'subqueries' in result and isinstance(result['subqueries'], list)
    assert 'query_is_ambiguous' in result and isinstance(result['query_is_ambiguous'], bool)
