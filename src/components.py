import warnings

try:
    from langchain_mistralai.chat_models import ChatMistralAI
    _HAS_MISTRAL_LANGCHAIN = True
except Exception:
    ChatMistralAI = None
    _HAS_MISTRAL_LANGCHAIN = False

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

try:
    from langchain.storage import InMemoryStore
except ModuleNotFoundError:
    from langchain_core.stores import InMemoryStore

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever, ContextualCompressionRetriever
except ModuleNotFoundError:
    from langchain_classic.retrievers import ParentDocumentRetriever, EnsembleRetriever, ContextualCompressionRetriever

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None

from . import config  # Import config for paths and settings

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# --- Initialize Components ---

# LLM
print(f"Initializing LLM: {config.LLM_MODEL_NAME} via GitHub Models")
if _HAS_MISTRAL_LANGCHAIN:
    llm = ChatMistralAI(
        model=config.LLM_MODEL_NAME,
        mistral_api_key=config.GITHUB_TOKEN,
        endpoint=config.GITHUB_MODELS_ENDPOINT,
        temperature=0.15,
    )
    print("LLM Initialized with ChatMistralAI.")
else:
    llm = ChatOpenAI(
        model=config.LLM_MODEL_NAME,
        openai_api_key=config.GITHUB_TOKEN,
        openai_api_base=config.GITHUB_MODELS_ENDPOINT,
        temperature=0.15,
    )
    print("LLM Initialized with ChatOpenAI fallback (langchain_mistralai not installed).")

# Embeddings
print(f"Loading Embedding Model: {config.EMBEDDING_MODEL_NAME} via GitHub Models")
embedding_model = OpenAIEmbeddings(
    model=config.EMBEDDING_MODEL_NAME,
    openai_api_key=config.GITHUB_TOKEN,
    openai_api_base=config.EMBEDDING_MODEL_ENDPOINT,
)
print("Embedding Model Initialized.")

# Vector Store and Document Store
print(f"Initializing Vector Store from: {config.VECTORSTORE_PATH}")
vectorstore = Chroma(
    persist_directory=config.VECTORSTORE_PATH,
    embedding_function=embedding_model,
)
store = InMemoryStore()
print("Vector Store and Document Store Initialized.")

# --- Web Search Component (Tavily) ---
print("Initializing Tavily Search API...")
try:
    if TavilyClient is not None and hasattr(config, "TAVILY_API_KEY") and config.TAVILY_API_KEY:
        tavily_client = TavilyClient(api_key=config.TAVILY_API_KEY)
        print("Tavily Search initialized successfully.")
    else:
        tavily_client = None
        print("WARNING: Tavily API key not found or tavily not installed. Web search capabilities disabled.")
except Exception as e:
    tavily_client = None
    print(f"WARNING: Tavily initialization failed: {e}. Web search capabilities disabled.")

print("All components initialized.")

# Chunking Strategy
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=config.PARENT_CHUNK_SIZE)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHILD_CHUNK_SIZE)

# Base Semantic Retriever
parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": config.VECTOR_K},
)

# --- Hybrid Search Components ---
def load_child_docs_from_storage(vs: Chroma) -> list[Document]:
    """
    WORKAROUND: Attempts to load all documents from the Chroma vectorstore
    assuming they represent the child documents needed for BM25.
    """
    print("Attempting to load child docs from vectorstore for BM25...")
    try:
        results = vs.get(include=["documents", "metadatas"])
        if results and results.get("documents"):
            child_docs_list = [
                Document(page_content=doc, metadata=meta or {})
                for doc, meta in zip(results["documents"], results.get("metadatas", []))
                if doc
            ]
            print(f"Loaded {len(child_docs_list)} documents from vectorstore for BM25.")
            return child_docs_list

        print("Could not retrieve documents from vectorstore for BM25.")
        return []
    except Exception as e:
        print(f"Error loading docs from vectorstore for BM25: {e}")
        return []

child_docs = load_child_docs_from_storage(vectorstore)

if child_docs:
    print("Initializing BM25 Retriever...")
    bm25_retriever = BM25Retriever.from_documents(
        documents=child_docs,
        k=config.BM25_K,
    )
    print("BM25 Retriever Initialized.")

    print("Initializing Ensemble Retriever...")
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": config.VECTOR_K})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.4, 0.6],
        search_kwargs={"k": config.ENSEMBLE_K},
    )
    print("Ensemble Retriever Initialized.")
    base_retriever_for_reranking = ensemble_retriever
else:
    print("BM25 Retriever skipped (could not load child docs). Using Parent Retriever as base for potential re-ranking.")
    base_retriever_for_reranking = parent_retriever

# --- Re-ranker Component (Optional) ---
USE_RERANKER = False

if USE_RERANKER:
    print("Initializing Re-ranker...")
    try:
        try:
            from langchain.retrievers.document_compressors import CrossEncoderReranker
        except ModuleNotFoundError:
            from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        cross_encoder_model = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        reranker_compressor = CrossEncoderReranker(
            model=cross_encoder_model,
            top_n=config.RERANK_TOP_N,
        )
        reranker_retriever = ContextualCompressionRetriever(
            base_compressor=reranker_compressor,
            base_retriever=base_retriever_for_reranking,
        )
        final_retriever = reranker_retriever
        print("Re-ranker Initialized. Using it for final retrieval.")
    except (ModuleNotFoundError, ImportError) as e:
        print(f"WARNING: Optional reranker deps missing: {e}. Falling back.")
        final_retriever = base_retriever_for_reranking
else:
    final_retriever = base_retriever_for_reranking
    print(f"Re-ranker Disabled. Using {'Ensemble' if child_docs else 'Parent'} Retriever as final.")

print("Retriever Components Initialized.")
# The 'final_retriever' variable now holds the retriever to be used by the agent node.