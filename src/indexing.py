import warnings # Import the warnings module
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.retrievers import ParentDocumentRetriever

from . import config # Import config for paths and settings
from .components import vectorstore, store, child_splitter, parent_splitter

import logging
logging.getLogger("pypdf").setLevel(logging.ERROR) # Set pypdf logging to ERROR level to suppress warnings ()
# Esta linea es para evitar warnings de pypdf que no afectan el funcionamiento, los cuales aparecen cuando un PDF tiene algo raro en su estructura.

# Reduce batch size to avoid hitting embedding model token limits per request
def index_documents(batch_size: int = 10): # Add batch_size parameter, reduced default
    """
    Loads PDFs, splits them using parent/child strategy, and indexes them
    into Chroma and the InMemoryStore in batches.

    Args:
        batch_size (int): Number of documents to process in each batch for indexing.
                          Adjust based on embedding model limits and memory.
                          A smaller size (e.g., 10-20) might be needed for large documents
                          or strict embedding API limits.
    """
    print(f"Loading documents from {config.DOCUMENTS_PATH}...(takes a while)")
    loader = PyPDFDirectoryLoader(config.DOCUMENTS_PATH)

    docs = loader.load()

    if not docs:
        print(f"No documents found in '{config.DOCUMENTS_PATH}'. Please add PDF files.")
        return False # Indicate failure or no action

    print(f"Loaded {len(docs)} documents.")

    # Check if vectorstore already has data
    try:
        if vectorstore._collection.count() > 0:
            print("Vectorstore already contains data. Skipping indexing.")
            # Note: ParentDocumentRetriever doesn't persist the InMemoryStore by default.
            # For full persistence, consider alternatives or manual save/load for 'store'.
            return True # Indicate success (already indexed)
    except Exception as e:
         print(f"Could not check vectorstore count (maybe first run?): {e}")
         # Continue with indexing

    print("Initializing ParentDocumentRetriever for indexing...")
    # Use the same components defined in components.py
    indexing_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store, # Use the shared InMemoryStore
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    print(f"Indexing documents in batches of {batch_size} (this may take a while)...")
    num_docs = len(docs)
    num_batches = (num_docs + batch_size - 1) // batch_size # Ceiling division
    current_batch_num = 0 # Initialize before the loop for error reporting

    try:
        for i in range(0, num_docs, batch_size):
            batch_docs = docs[i : i + batch_size]
            current_batch_num = (i // batch_size) + 1
            print(f"  Processing batch {current_batch_num}/{num_batches} ({len(batch_docs)} documents)...")

            # The add_documents method handles splitting and indexing for the current batch
            indexing_retriever.add_documents(batch_docs, ids=None) # Langchain manages IDs

            print(f"  Batch {current_batch_num} indexed.")
            # Optional: Add a small delay between batches if hitting rate limits
            # import time
            # time.sleep(1)

        # Chroma persists automatically due to 'persist_directory' after all batches
        print("Indexing complete. Vectorstore persisted.")
        return True # Indicate success
    except Exception as e:
        # Ensure current_batch_num is defined even if error happens before first batch loop iteration
        batch_info = f"batch {current_batch_num}/{num_batches}" if current_batch_num > 0 else "initialization or first batch"
        print(f"Error during indexing {batch_info}: {e}")
        print("Indexing stopped due to error.")
        # Consider if partial indexing is acceptable or if cleanup is needed
        return False # Indicate failure