import os
from dotenv import load_dotenv

load_dotenv()

# API Keys and Endpoints for GitHub Models
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "YOUR_GITHUB_TOKEN") # Use GITHUB_TOKEN env var
GITHUB_MODELS_ENDPOINT = "https://models.github.ai/inference" # Specific endpoint
LLM_MODEL_NAME = "mistral-ai/Mistral-Large-2411" # Specific model name from GitHub Models
# API keys and endpoints for Mistral models (another option)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "YOUR_MISTRAL_API_KEY") # Use MISTRAL_API_KEY env var
LLM_MISTRAL_NAME = "mistral-large-latest" # It is the same as Mistral-Large-2411 but with a different name in the API

# Judge LLM
JUDGE_ENDPOINT=os.getenv("JUDGE_ENDPOINT") 
JUDGE_KEY=os.getenv("JUDGE_KEY")
#embedding evaluation
EMBEDDING_EVAL_KEY=os.getenv("EMBEDDING_EVAL_KEY")
EMBEDDING_EVAL_ENDPOINT=os.getenv("EMBEDDING_EVAL_ENDPOINT")


# Embedding Model Configuration
# EMBEDDING_MODEL_NAME = "nvidia/NV-Embed-v2" # Opción 1 
# EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-Qwen2-7B-instruct" # Opción 2 
# EMBEDDING_MODEL_NAME = "text-embedding-3-large" # Opción OpenAI via GitHub Models 
# EMBEDDING_DEVICE = 'cpu' # Cambiar a 'cuda' si aplica
# NORMALIZE_EMBEDDINGS = True4

# Como mi portatil no tiene capacidad de ejecutar estos modelos en local, necesito alguno via API
EMBEDDING_MODEL_NAME = "text-embedding-3-large" # Opción OpenAI via GitHub Models
EMBEDDING_MODEL_ENDPOINT = "https://models.inference.ai.azure.com" # Specific endpoint for OpenAI models

# Paths and Directories
VECTORSTORE_PATH = "./chroma_db_diabetes"
DOCUMENTS_PATH = "./diabetes_docs" # Directorio con los PDFs a indexar

# Tavily API Key 
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "YOUR_TAVILY_API_KEY") # Use TAVILY_API_KEY env var
WEB_SEARCH_MAX_RESULTS = 5

# Chunking and Retrieval Parameters
PARENT_CHUNK_SIZE = 2000 # Tamaño más grande para el contexto del ParentDocumentRetriever
CHILD_CHUNK_SIZE = 400 # Tamaño pequeño para búsqueda eficiente
BM25_K = 50 # Número de documentos a recuperar por BM25
VECTOR_K = 50 # Número de documentos a recuperar por Vector Search
ENSEMBLE_K = 50 # Número de documentos finales tras Ensemble/Hybrid search
RERANK_TOP_N = 50 # Número de documentos tras el re-ranking

# Graph Configuration
RECURSION_LIMIT = 5