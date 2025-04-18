import warnings
from langchain_mistralai.chat_models import ChatMistralAI
# from langchain_huggingface import HuggingFaceEmbeddings   # Uncomment if using HuggingFace for embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document 

from tavily import TavilyClient

import config

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')

# --- Initialize Components ---

# LLM
print(f"Initializing LLM: {config.LLM_MODEL_NAME} via GitHub Models")
llm = ChatMistralAI(
    model=config.LLM_MODEL_NAME,
    mistral_api_key=config.GITHUB_TOKEN,
    endpoint=config.GITHUB_MODELS_ENDPOINT,
    temperature=0.15, # A low temperature for more deterministic responses
)
print("LLM Initialized.")

# Embeddings Model. Change to HuggingFaceEmbeddings if using HuggingFace models
print(f"Loading Embedding Model: {config.EMBEDDING_MODEL_NAME} via GitHub Models")
embedding_model = OpenAIEmbeddings(
    model=config.EMBEDDING_MODEL_NAME,
    openai_api_key=config.GITHUB_TOKEN,
    openai_api_base=config.EMBEDDING_MODEL_ENDPOINT,
    # Add other parameters like 'chunk_size' if needed, check OpenAIEmbeddings documentation
)
print("Embedding Model Initialized.")

# Vector Store and Document Store
print(f"Initializing Vector Store from: {config.VECTORSTORE_PATH}")
vectorstore = Chroma(
    persist_directory=config.VECTORSTORE_PATH,
    embedding_function=embedding_model
)
store = InMemoryStore() 
print("Vector Store and Document Store Initialized.")

# --- Web Search Component (Tavily) ---
print("Initializing Tavily Search API...")
try:
    
    # Inicializamos si existe la clave API
    if hasattr(config, 'TAVILY_API_KEY') and config.TAVILY_API_KEY:
        tavily_client = TavilyClient(api_key=config.TAVILY_API_KEY)
        print("Tavily Search initialized successfully.")
    else:
        tavily_client = None
        print("WARNING: Tavily API key not found. Web search capabilities disabled.")
except ImportError:
    tavily_client = None
    print("WARNING: tavily-python package not installed. Web search capabilities disabled.")

print("All components initialized.")

# Chunking Strategy
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=config.PARENT_CHUNK_SIZE)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHILD_CHUNK_SIZE)

# Base Semantic Retriever: ParentDocumentRetriever
parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store, # The InMemoryStore for parent docs
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": config.VECTOR_K}
)

# --- Hybrid Search Components ---
# Function to load child documents (WORKAROUND: Fetches from vectorstore)
def load_child_docs_from_storage(vs: Chroma) -> list[Document]:
    """
    WORKAROUND: Attempts to load all documents from the Chroma vectorstore
    assuming they represent the child documents needed for BM25.

    IDEAL FIX: Modify indexing.py to save child docs explicitly and load them here.
    """
    print("Attempting to load child docs from vectorstore for BM25...")
    try:
        # Fetch all documents. Note: This might be inefficient for large stores.
        # The exact method might vary slightly based on Chroma version.
        # We need Document objects with page_content.
        results = vs.get(include=["documents", "metadatas"]) # Fetch documents and metadata
        if results and results.get("documents"):
            child_docs_list = [
                Document(page_content=doc, metadata=meta or {})
                for doc, meta in zip(results["documents"], results.get("metadatas", []))
                if doc # Ensure content exists
            ]
            print(f"Loaded {len(child_docs_list)} documents from vectorstore for BM25.")
            return child_docs_list
        else:
            print("Could not retrieve documents from vectorstore for BM25.")
            return []
    except Exception as e:
        print(f"Error loading docs from vectorstore for BM25: {e}")
        return []

# Attempt to initialize BM25 using the workaround
child_docs = load_child_docs_from_storage(vectorstore)

if child_docs:
    print("Initializing BM25 Retriever...")
    bm25_retriever = BM25Retriever.from_documents(
        documents=child_docs, # Use documents loaded from vectorstore
        k=config.BM25_K
    )
    print("BM25 Retriever Initialized.")

    print("Initializing Ensemble Retriever...")
    # Semantic retriever component for the ensemble
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": config.VECTOR_K})

    # Ensemble Retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever], # Use BM25 and pure semantic search
        weights=[0.4, 0.6], # Adjust weights as needed
        search_kwargs={"k": config.ENSEMBLE_K} # How many results the ensemble should return
    )
    print("Ensemble Retriever Initialized.")
    # The base retriever for potential re-ranking is now the Ensemble Retriever
    base_retriever_for_reranking = ensemble_retriever
else:
    print("BM25 Retriever skipped (could not load child docs). Using Parent Retriever as base for potential re-ranking.")
    # Fallback if BM25/Ensemble cannot be initialized
    # Using ParentDocumentRetriever ensures we still get context from parent docs
    base_retriever_for_reranking = parent_retriever


# --- Re-ranker Component (Optional, set USE_RERANKER=True to enable) ---
USE_RERANKER = False # Set to True to enable the re-ranker below

if USE_RERANKER:
    print("Initializing Re-ranker...")
    # Example with CrossEncoder
    cross_encoder_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2") # Choose a model
    reranker_compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=config.RERANK_TOP_N)
    # Apply re-ranking ON TOP of the chosen base retriever (Ensemble or Parent)
    reranker_retriever = ContextualCompressionRetriever(
         base_compressor=reranker_compressor,
         base_retriever=base_retriever_for_reranking
    )
    final_retriever = reranker_retriever # Use re-ranker if enabled
    print("Re-ranker Initialized. Using it for final retrieval.")
else:
    # If re-ranker is not used, the final retriever is the base one chosen earlier
    final_retriever = base_retriever_for_reranking
    print(f"Re-ranker Disabled. Using {'Ensemble' if child_docs else 'Parent'} Retriever as final.")


print("Retriever Components Initialized.")
# The 'final_retriever' variable now holds the retriever to be used by the agent node.