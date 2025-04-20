import streamlit as st
import os
import io
import contextlib
from typing import List, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError 

# --- Import necessary components from your project ---
# Ensure these imports work correctly based on your project structure
try:
    from src import config # Import config for paths and settings
    from src.components import vectorstore # Import vectorstore for checking count
    from src.indexing import index_documents # Import indexing function
    from src.graph import app # Import the compiled graph app
    from src.agent import AgentState # Import AgentState if needed for type hints
except ImportError as e:
    st.error(f"Error importing project modules: {e}. Make sure the script is run from the project root or paths are configured correctly.")
    st.stop()

# --- Constants ---
ASSISTANT_AVATAR = "🤖"
USER_AVATAR = "👤"
UNKNOWN_CONFIDENCE = "N/A"

# --- Helper Function to Run the Agent and Capture Logs ---
def run_agent_with_logging(query: str, chat_history: List[BaseMessage]) -> Tuple[str, List[BaseMessage], str, str]:
    """
    Runs the Agentic RAG graph, captures logs, and returns results.

    Returns:
        Tuple[str, List[BaseMessage], str, str]: final_answer, new_history, confidence, logs
    """
    inputs = {
        "original_query": query,
        "chat_history": chat_history
    }
    run_config = RunnableConfig(recursion_limit=config.RECURSION_LIMIT)
    log_stream = io.StringIO()
    final_answer = "Error: No se generó respuesta."
    confidence = UNKNOWN_CONFIDENCE
    new_history = chat_history # Default to old history in case of error
    final_state = {}

    try:
        # Redirect stdout to capture print statements
        with contextlib.redirect_stdout(log_stream):
            print("--- Starting Agent Execution ---")
            final_state = app.invoke(inputs, config=run_config)
            print("--- Agent Execution Finished ---")

        # Extract results from the final state
        final_answer = final_state.get("final_answer", "Error: No se pudo obtener la respuesta final del estado.")
        confidence_raw = final_state.get("generation_confidence") # Might be None

        # Determine confidence string
        if confidence_raw == "high":
            confidence = "Alta"
        elif confidence_raw == "low":
            confidence = "Baja"
        elif final_state.get("error_message"):
             confidence = "Error" # Indicate error via confidence
             final_answer = f"Error: {final_state.get('error_message', 'Unknown error')}"
        else:
             confidence = UNKNOWN_CONFIDENCE # If confidence wasn't set

        # Update history safely
        new_history = chat_history + [
            HumanMessage(content=query),
            AIMessage(content=final_answer) # Store the potentially error message in history too
        ]

    except GraphRecursionError as e: # Catch specific recursion error
        st.error(f"Error: Límite de recursión del grafo alcanzado: {e}")
        final_answer = "Lo siento, el proceso se complicó y no pude completar tu solicitud. Intenta reformular la pregunta o simplificarla."
        confidence = "Error"
        new_history = chat_history + [
            HumanMessage(content=query),
            AIMessage(content=final_answer)
        ]
        print(f"\n--- GRAPH RECURSION ERROR ---", file=log_stream)
        print(f"{e}", file=log_stream)
        import traceback
        traceback.print_exc(file=log_stream)

    except Exception as e:
        st.error(f"Error during agent execution: {e}")
        final_answer = f"Lo siento, ocurrió un error grave al procesar tu solicitud: {e}"
        confidence = "Error"
        # Ensure history reflects the error message
        new_history = chat_history + [
            HumanMessage(content=query),
            AIMessage(content=final_answer)
        ]
        # Capture the exception in the logs as well
        print(f"\n--- EXCEPTION CAUGHT ---", file=log_stream)
        import traceback
        traceback.print_exc(file=log_stream)

    logs = log_stream.getvalue()
    return final_answer, new_history, confidence, logs

# --- Streamlit App ---

st.set_page_config(page_title="Asistente Diabetes", page_icon="🩸")
st.title("Asistente de Diabetes")
st.caption("Consulta información sobre diabetes basada en documentos médicos.")

# --- Initialization and Indexing ---
# Run this only once per session
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.indexing_done = False
    st.session_state.error_message = None

if not st.session_state.initialized:
    with st.status("Inicializando sistema...", expanded=True) as status:
        try:
            st.write(f"Verificando directorio de documentos: '{config.DOCUMENTS_PATH}'...")
            if not os.path.exists(config.DOCUMENTS_PATH):
                os.makedirs(config.DOCUMENTS_PATH)
                st.warning(f"Directorio '{config.DOCUMENTS_PATH}' creado. Por favor, añade documentos PDF y refresca la página.")
                st.session_state.error_message = "Directorio de documentos creado. Añade PDFs."
                st.stop() # Stop execution until user adds files

            pdf_files = [f for f in os.listdir(config.DOCUMENTS_PATH) if f.lower().endswith('.pdf')]
            if not pdf_files:
                st.warning(f"No se encontraron archivos PDF en '{config.DOCUMENTS_PATH}'. El asistente no tendrá documentos para consultar.")
                # Allow proceeding without documents, but RAG won't work well
                st.session_state.indexing_done = True # Mark as "done" as there's nothing to index
            else:
                st.write(f"Encontrados {len(pdf_files)} archivos PDF.")
                st.write("Verificando base de datos vectorial...")
                try:
                    count = vectorstore._collection.count()
                    if count == 0:
                        st.write("Base de datos vacía. Indexando documentos (esto puede tardar)...")
                        # Capture indexing logs
                        log_stream_index = io.StringIO()
                        with contextlib.redirect_stdout(log_stream_index):
                            success = index_documents() # Use default batch size or adjust if needed
                        st.text_area("Log de Indexación", log_stream_index.getvalue(), height=150)
                        if success:
                            st.write("Indexación completada.")
                            st.session_state.indexing_done = True
                        else:
                            st.error("La indexación falló. Revisa los logs.")
                            st.session_state.error_message = "La indexación falló."
                            st.stop()
                    else:
                        st.write(f"Base de datos vectorial contiene {count} entradas. Saltando indexación.")
                        st.session_state.indexing_done = True
                except Exception as e:
                    st.error(f"Error al verificar/indexar la base de datos: {e}")
                    st.session_state.error_message = f"Error de base de datos: {e}"
                    st.stop()

            if st.session_state.indexing_done:
                status.update(label="Inicialización completa.", state="complete", expanded=False)
                st.session_state.initialized = True
            else:
                 status.update(label="Inicialización fallida.", state="error", expanded=True)

        except Exception as e:
            st.error(f"Error fatal durante la inicialización: {e}")
            st.session_state.error_message = f"Error fatal: {e}"
            st.stop()

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [] # Store dicts like {"role": "user/assistant", "content": ..., "confidence": ..., "logs": ...}
if "langchain_history" not in st.session_state:
     st.session_state.langchain_history = [] # Store Langchain BaseMessage objects

# Display previous messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        # Display confidence and logs expander only for assistant messages
        if message["role"] == "assistant":
            confidence = message.get("confidence", UNKNOWN_CONFIDENCE)
            st.markdown(f"*Confianza en la respuesta: {confidence}*")
            if message.get("logs"):
                with st.expander("Ver Logs de Ejecución"):
                    st.text(message["logs"])

# --- Chat Input ---
if prompt := st.chat_input("Escribe tu consulta sobre diabetes..."):
    # Add user message to Streamlit state
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # Run the agent and get response
    with st.spinner("Pensando..."):
        final_answer, new_langchain_history, confidence, logs = run_agent_with_logging(
            prompt,
            st.session_state.langchain_history
        )

    # Update Langchain history
    st.session_state.langchain_history = new_langchain_history

    # Add assistant response to Streamlit state
    assistant_message = {
        "role": "assistant",
        "content": final_answer,
        "confidence": confidence,
        "logs": logs
    }
    st.session_state.messages.append(assistant_message)

    # Display assistant message
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.markdown(final_answer)
        st.markdown(f"*Confianza en la respuesta: {confidence}*")
        with st.expander("Ver Logs de Ejecución"):
            st.text(logs)

# Display error message if initialization failed
if not st.session_state.initialized and st.session_state.error_message:
    st.error(f"El sistema no pudo inicializarse correctamente: {st.session_state.error_message}")