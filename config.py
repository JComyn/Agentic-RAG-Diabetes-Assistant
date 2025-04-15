import os
from dotenv import load_dotenv

load_dotenv()

# --- Constants based on TFG Design ---
# API Keys and Endpoints for GitHub Models
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "YOUR_GITHUB_TOKEN") # Use GITHUB_TOKEN env var
GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com" # Specific endpoint
LLM_MODEL_NAME = "Mistral-Large-2411" # Specific model name from GitHub Models

# Embedding Model Configuration
# EMBEDDING_MODEL_NAME = "nvidia/NV-Embed-v2" # Opción 1 
# EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-Qwen2-7B-instruct" # Opción 2 
# EMBEDDING_MODEL_NAME = "text-embedding-3-large" # Opción OpenAI via GitHub Models 
# EMBEDDING_DEVICE = 'cpu' # Cambiar a 'cuda' si aplica
# NORMALIZE_EMBEDDINGS = True4

# Como mi portatil no tiene capacidad de ejecutar estos modelos en local, necesito alguno via API
EMBEDDING_MODEL_NAME = "text-embedding-3-large" # Opción OpenAI via GitHub Models

# Paths and Directories
VECTORSTORE_PATH = "./chroma_db_diabetes"
DOCUMENTS_PATH = "./diabetes_docs" # Directorio con los PDFs del TFG Sec 3.1

# Chunking and Retrieval Parameters
PARENT_CHUNK_SIZE = 2000 # Tamaño más grande para el contexto del ParentDocumentRetriever
CHILD_CHUNK_SIZE = 400 # Tamaño pequeño para búsqueda eficiente
BM25_K = 50 # Número de documentos a recuperar por BM25
VECTOR_K = 50 # Número de documentos a recuperar por Vector Search
ENSEMBLE_K = 50 # Número de documentos finales tras Ensemble/Hybrid search
RERANK_TOP_N = 50 # Número de documentos tras el re-ranking

# Graph Configuration
RECURSION_LIMIT = 5