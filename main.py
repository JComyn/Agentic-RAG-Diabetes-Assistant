import os
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError # Add this import at the top

from src import config # Import config for paths and settings
from src.components import vectorstore # Import vectorstore for checking count
from src.indexing import index_documents # Import indexing function
from src.graph import app # Import the compiled graph app


# --- Function to run the chat interaction ---
def run_chat(query: str, chat_history: List[BaseMessage] = []):
    """
    Runs the Agentic RAG graph for a given query and chat history.
    """
    inputs = {
        "original_query": query,
        "chat_history": chat_history
    }

    # Configuration for the graph run (e.g., for LangSmith tracing)
    run_config = RunnableConfig(recursion_limit=config.RECURSION_LIMIT)

    try:
        # Invoke the graph
        final_state = app.invoke(inputs, config=run_config)

        # Get the final answer
        final_answer = final_state.get("final_answer", "Error: No se generó respuesta.")

        # Update history
        new_history = chat_history + [
            HumanMessage(content=query),
            AIMessage(content=final_answer)
        ]

        return final_answer, new_history

    except GraphRecursionError as e: # Catch specific recursion error
        print(f"Error: Graph recursion limit reached: {e}")
        error_message = "Lo siento, el proceso se complicó y no pude completar tu solicitud. Intenta reformular la pregunta o simplificarla."
        # Update history with specific error
        new_history = chat_history + [
            HumanMessage(content=query),
            AIMessage(content=error_message)
        ]
        return error_message, new_history
    except Exception as e: # Catch other general exceptions
        print(f"Error during chat execution: {e}")
        error_message = "Lo siento, ocurrió un error inesperado al procesar tu solicitud."
        # Update history with error
        new_history = chat_history + [
            HumanMessage(content=query),
            AIMessage(content=error_message)
        ]
        return error_message, new_history


# --- Main Execution ---
if __name__ == "__main__":
    # --- Initial Setup: Check/Create Document Directory and Index ---
    if not os.path.exists(config.DOCUMENTS_PATH):
        os.makedirs(config.DOCUMENTS_PATH)
        print(f"Directory '{config.DOCUMENTS_PATH}' created. Please add PDF documents here and restart.")
        exit() # Exit if directory was just created

    # Check for PDF files before attempting to index
    pdf_files = [f for f in os.listdir(config.DOCUMENTS_PATH) if f.lower().endswith('.pdf')]
    if not pdf_files:
         print(f"No PDF files found in '{config.DOCUMENTS_PATH}'. Add documents to index.")
         # Decide if you want to proceed without documents or exit
         # exit()
    else:
        # Attempt to index documents if the vectorstore seems empty
        try:
             count = vectorstore._collection.count()
             if count == 0:
                 print("Vectorstore appears empty. Attempting to index documents...")
                 index_documents()
             else:
                 print(f"Vectorstore contains {count} entries. Skipping indexing.")
        except Exception as e:
             print(f"Could not check vectorstore or index documents (Error: {e}). Proceeding without indexing check.")
             # Optionally try indexing anyway or handle the error
             # index_documents()


    # --- Interactive Chat Loop ---
    print("\n--- Asistente de Diabetes ---")
    print("Escribe tu consulta sobre diabetes o 'salir' para terminar.")

    history: List[BaseMessage] = []
    while True:
        user_input = input("Tú: ")
        if user_input.lower() == 'salir':
            break
        if not user_input.strip():
            continue

        # Execute the Agentic RAG flow
        answer, history = run_chat(user_input, history)

        print(f"Asistente: {answer}")

    print("\n--- Fin de la sesión ---")
    print("Gracias por usar el asistente de diabetes.")
    