from typing import List, TypedDict, Annotated, Sequence, Literal
import json
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import END # Import END for graph termination

# Import necessary components
from .components import (
    llm,
    tavily_client,
    final_retriever as retriever, # Use the final retriever configured in components.py
    USE_RERANKER
)

from .config import (
    RERANK_TOP_N,
    WEB_SEARCH_MAX_RESULTS
)

# --- Agent State Definition ---
ValidationStatus = Literal['sufficient', 'insufficient_reretrieve', 'insufficient_clarify', 'error']

class AgentState(TypedDict):
    original_query: str
    chat_history: Sequence[BaseMessage]
    # Agent outputs
    transformed_query: str | None
    retrieved_docs: List[Document] | None # Docs before validation/re-ranking
    validated_docs: List[Document] | None # Docs after validation/re-ranking
    final_answer: str | None
    clarification_question: str | None # For asking the user
    # Control flow fields
    validation_status: ValidationStatus | None
    generation_confidence: Literal['high', 'low', None]
    error_message: str | None # To store errors during execution
    # Counter for re-retrieval attempts
    reretrieval_attempts: int

# --- Agent Nodes (Functions for LangGraph) ---

# 1. Query Transformer Agent
def transform_query_node(state: AgentState, config: RunnableConfig):
    """Transforms the user query for better retrieval."""
    print("--- Node: Transform Query ---")
    original_query = state["original_query"]
    chat_history = state.get("chat_history", [])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un experto en diabetes y en reformular preguntas para mejorar la búsqueda de información. Reescribe la siguiente consulta del usuario para optimizarla para una base de datos vectorial. Considera el historial. Responde SÓLO con la consulta reescrita."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Consulta original: {query}")
    ])
    chain = prompt | llm | StrOutputParser()
    try:
        transformed_query = chain.invoke({"query": original_query, "chat_history": chat_history}, config=config) 
        print(f"Original Query: {original_query}")
        print(f"Transformed Query: {transformed_query}")
        # Reset re-retrieval counter when transforming query
        return {"transformed_query": transformed_query, "error_message": None, "reretrieval_attempts": 0}
    except Exception as e:
        print(f"Error in transform_query_node: {e}")
        return {"transformed_query": original_query, "error_message": f"Error transforming query: {e}", "reretrieval_attempts": 0} # Fallback

# 2. Document Retriever Agent
def retrieve_documents_node(state: AgentState, config: RunnableConfig):
    """Retrieves documents using the configured retriever."""
    print("--- Node: Retrieve Documents ---")
    query = state["transformed_query"] or state["original_query"]
    print(f"Retrieving documents for query: {query}")
    try:
        # Usar retriever configurado en components.py
        retrieved_docs = retriever.invoke(query)
        print(f"Retrieved {len(retrieved_docs)} documents.")
        return {"retrieved_docs": retrieved_docs, "error_message": None}
    except Exception as e:
        print(f"Error in retrieve_documents_node: {e}")
        return {"retrieved_docs": [], "error_message": f"Error retrieving documents: {e}"}

# 3. Document Validator Agent (LLM Enhanced)
def validate_documents_node(state: AgentState, config: RunnableConfig):
    """Validates documents using LLM assessment."""
    print("--- Node: Validate Documents ---")
    retrieved_docs = state.get("retrieved_docs")
    query = state["transformed_query"] or state["original_query"]
    current_attempts = state.get("reretrieval_attempts", 0)

    if not retrieved_docs:
        print("Validation failed: No documents retrieved.")
        return {
            "validated_docs": [],
            "validation_status": "insufficient_reretrieve",
            "error_message": "No documents to validate.",
            "reretrieval_attempts": current_attempts + 1
        }

    try:
        if USE_RERANKER:
            docs_to_validate = retrieved_docs
            print("Using " f"{len(docs_to_validate)} documents for validation (re-ranking applied)")
        else:
            docs_to_validate = retrieved_docs[:RERANK_TOP_N] # Limit to top N for validation (to avoid exceeding token limits)
            print("Using " f"{len(docs_to_validate)} documents for validation (no re-ranking applied)")
            
        if not docs_to_validate:
            print("Validation failed: No documents to validate.")
            return {
                "validated_docs": [],
                "validation_status": "insufficient_reretrieve",
                "error_message": "No documents to validate.",
                "reretrieval_attempts": current_attempts + 1
            }

        # --- LLM Assessment ---
        print("Performing LLM validation assessment...")
        context_for_validation = "\n\n---\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs_to_validate)])

        # Construct the system message content directly using f-string
        system_message_content = f"""Eres un asistente experto en validación de información sobre diabetes. Tu tarea es evaluar si los documentos proporcionados son RELEVANTES y ofrecen una BUENA BASE para responder a la consulta del usuario. No exijas que la respuesta sea exhaustiva, solo que sea útil y esté bien fundamentada en los documentos.
La consulta es: "{query}"

Documentos proporcionados:
{context_for_validation}

Analiza los documentos en relación a la consulta y responde ÚNICAMENTE con un objeto JSON que contenga:
1. "decision": Una de las siguientes opciones: "sufficient", "insufficient_reretrieve", "insufficient_clarify".
    - "sufficient": Si los documentos tratan sobre el tema principal de la consulta y contienen información que permite formular una respuesta útil.
    - "insufficient_reretrieve": Si la MAYORÍA de los documentos son irrelevantes o solo tocan el tema de forma muy tangencial, haciendo difícil construir una respuesta. Una nueva búsqueda podría ser necesaria.
    - "insufficient_clarify": Si los documentos son irrelevantes PORQUE la consulta es demasiado ambigua para ser respondida adecuadamente, necesitando aclaración del usuario.
2. "reasoning": Una breve explicación (1-2 frases) de tu decisión.
3. "clarification_question" (OPCIONAL): Si la decisión es "insufficient_clarify", formula una pregunta específica al usuario para obtener la información necesaria.

Ejemplo de respuesta JSON:
{{
  "decision": "sufficient",
  "reasoning": "Los documentos explican qué es la diabetes tipo 2 y mencionan sus síntomas principales, lo cual es suficiente para una respuesta inicial."
}}
O
{{
  "decision": "insufficient_reretrieve",
  "reasoning": "Los documentos hablan sobre dieta, pero no específicamente sobre el índice glucémico mencionado en la consulta. Una nueva búsqueda podría ser más específica."
}}
O
{{
  "decision": "insufficient_clarify",
  "reasoning": "La consulta 'efectos' es muy amplia. No está claro si se refiere a efectos a corto o largo plazo, o en qué sistema del cuerpo.",
  "clarification_question": "¿Podrías especificar a qué tipo de efectos de la diabetes te refieres (a corto plazo, a largo plazo, sobre el corazón, etc.)?"
}}
"""
        # Create the list of messages directly
        messages = [
            SystemMessage(content=system_message_content),
            HumanMessage(content="Evalúa los documentos proporcionados y responde en formato JSON según las instrucciones.")
        ]

        # Define the parser for the JSON output
        parser = JsonOutputParser()
        parser_with_fallback = parser.with_fallbacks([RunnableLambda(lambda x: {"decision": "error", "reasoning": f"Fallback: Could not parse LLM output: {x}"})])

        # Create the chain starting from the messages list
        validation_chain = RunnableLambda(lambda _: messages) | llm | parser_with_fallback

        # Invoke the chain 
        llm_assessment = validation_chain.invoke({}, config=config) 

        print(f"LLM Validation Assessment: {llm_assessment}")

        decision = llm_assessment.get("decision", "error")
        clarification_q = llm_assessment.get("clarification_question")

        if decision == "insufficient_reretrieve" and tavily_client:
            print("Local documents insufficient - attempting web search via Tavily...")
            try:
                # Perform web search
                search_results = tavily_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=WEB_SEARCH_MAX_RESULTS
                )
                
                # Convert search results to Document objects
                web_docs = []
                for result in search_results.get('results', []):
                    content = f"TITLE: {result.get('title', 'No title')}\n"
                    content += f"CONTENT: {result.get('content', 'No content')}\n"
                    content += f"URL: {result.get('url', 'No URL')}"
                    
                    web_docs.append(Document(
                        page_content=content,
                        metadata={
                            "source": "web_search",
                            "url": result.get('url', 'No URL'),
                            "title": result.get('title', 'No title'),
                            "score": result.get('score', 0)
                        }
                    ))
                
                if web_docs:
                    print(f"Retrieved {len(web_docs)} documents from web search.")
                    
                    # Combine with existing documents
                    combined_docs = docs_to_validate + web_docs
                    
                    # Re-assess with LLM using the combined documents
                    context_for_validation = "\n\n---\n\n".join([
                        f"Document {i+1} ({'Web' if doc.metadata.get('source') == 'web_search' else 'Local'}):\n{doc.page_content}" 
                        for i, doc in enumerate(combined_docs)
                    ])
                    
                    # Update system message to include info about web sources
                    system_message_content = f"""Eres un asistente experto en validación de información sobre diabetes. Tu tarea es evaluar si los documentos proporcionados son RELEVANTES y ofrecen una BUENA BASE para responder a la consulta del usuario.
La consulta es: "{query}"

Documentos proporcionados (incluyen fuentes locales y web):
{context_for_validation}

Analiza los documentos en relación a la consulta y responde ÚNICAMENTE con un objeto JSON que contenga:
1. "decision": Una de las siguientes opciones: "sufficient", "insufficient_reretrieve", "insufficient_clarify".
    - "sufficient": Si los documentos tratan sobre el tema principal de la consulta y contienen información que permite formular una respuesta útil.
    - "insufficient_reretrieve": Si la MAYORÍA de los documentos son irrelevantes o solo tocan el tema de forma muy tangencial, haciendo difícil construir una respuesta. Una nueva búsqueda podría ser necesaria.
    - "insufficient_clarify": Si los documentos son irrelevantes PORQUE la consulta es demasiado ambigua para ser respondida adecuadamente, necesitando aclaración del usuario.
2. "reasoning": Una breve explicación (1-2 frases) de tu decisión.
3. "clarification_question" (OPCIONAL): Si la decisión es "insufficient_clarify", formula una pregunta específica al usuario para obtener la información necesaria.
"""
                    # Create updated messages
                    web_validation_messages = [
                        SystemMessage(content=system_message_content),
                        HumanMessage(content="Evalúa los documentos proporcionados y responde en formato JSON según las instrucciones.")
                    ]
                    
                    # Create and invoke the updated chain
                    web_validation_chain = RunnableLambda(lambda _: web_validation_messages) | llm | parser_with_fallback
                    web_assessment = web_validation_chain.invoke({}, config=config)
                    
                    print(f"Web-enhanced LLM Assessment: {web_assessment}")
                    
                    # If web search improved the results to sufficient, use the combined docs
                    if web_assessment.get("decision") == "sufficient":
                        print("Web search provided sufficient information!")
                        return {
                            "validated_docs": combined_docs,
                            "validation_status": "sufficient",
                            "error_message": None,
                            "reretrieval_attempts": current_attempts
                        }
                else:
                    print("Web search returned no results.")
            except Exception as web_e:
                print(f"Error during web search: {web_e}")
                # Continue with normal flow if web search fails

        # Map LLM decision to ValidationStatus type 
        validation_status: ValidationStatus
        next_attempts = current_attempts
        if decision == "sufficient":
            validation_status = "sufficient"
        elif decision == "insufficient_reretrieve":
            validation_status = "insufficient_reretrieve"
            next_attempts = current_attempts + 1
        elif decision == "insufficient_clarify":
            validation_status = "insufficient_clarify"

        return {
            "validated_docs": docs_to_validate if validation_status == "sufficient" else [], # Only pass docs if sufficient
            "validation_status": validation_status,
            "clarification_question": clarification_q if validation_status == "insufficient_clarify" else None,
            "error_message": f"LLM validation failed: {llm_assessment.get('reasoning')}" if validation_status == "error" else None,
            "reretrieval_attempts": next_attempts # Return updated attempt count
        }

    except Exception as e:
        print(f"Error in validate_documents_node (Outer Try/Except): {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        # Fallback if anything else goes wrong
        return {
            "validated_docs": [],
            "validation_status": "error",
            "error_message": f"Error during document validation: {e}",
            "reretrieval_attempts": current_attempts # Keep attempts as they were before this failed node run
        }


# --- Confidence Assessment Chain (Helper Function) ---

confidence_prompt = ChatPromptTemplate.from_messages([
    ("system", """Eres un evaluador experto de respuestas de asistentes de IA sobre diabetes. Tu tarea es determinar si la respuesta proporcionada es CONFIABLE y responde DIRECTAMENTE a la pregunta del usuario. No tienes acceso al contexto original, solo a la pregunta y la respuesta. Responde únicamente con "high" o "low".

    - Responde "high" si la respuesta aborda directamente la pregunta del usuario y parece informativa y segura (evita frases como "no estoy seguro", "podría ser", "quizás").
    - Responde "low" si la respuesta es evasiva, genérica (como un simple saludo), indica incapacidad para responder ("no puedo responder", "no encontré información"), o no parece relacionada directamente con la pregunta.

    Pregunta del Usuario:
    {query}

    Respuesta Generada:
    {answer}
    """),
    ("human", "Evalúa la confianza de la Respuesta Generada para la Pregunta del Usuario. Responde solo 'high' o 'low'.")
])

# Using the same LLM instance but could be a different one if needed
confidence_chain = confidence_prompt | llm | StrOutputParser()

# 4. Answer Generation Agent
def generate_answer_node(state: AgentState, config: RunnableConfig): # Changed config_run to config
    """Generates the final answer based on validated documents and assesses its confidence."""
    print("--- Node: Generate Answer ---")
    query = state["original_query"] # Use original query for context and confidence check
    documents = state.get("validated_docs")
    chat_history = state.get("chat_history", [])

    if not documents:
        print("Generation Error: No validated documents available.")
        return {
            "final_answer": "Lo siento, hubo un problema interno y no pude procesar la información encontrada.",
            "generation_confidence": "low",
            "error_message": "Generation error: No validated documents provided."
        }

    context = "\n\n---\n\n".join([doc.page_content for doc in documents])
    # print(f"Context for generation (first 500 chars):\n{context[:500]}...") # Optional: uncomment for debugging

    generation_prompt = ChatPromptTemplate.from_messages([
         ("system", """Eres un asistente virtual experto en diabetes. Responde a la pregunta del usuario de forma clara, concisa y empática, basándote ESTRICTAMENTE en el contexto proporcionado. NO inventes información.

Contexto:
{context}
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}")
    ])

    generation_chain = generation_prompt | llm | StrOutputParser()

    final_answer = "Error: Fallo en la generación inicial." # Default
    confidence = "low" # Default confidence
    generation_error = None # Initialize error message

    try:
        print("Generating final answer...")
        input_dict = {"query": query, "chat_history": chat_history, "context": context}
        final_answer = generation_chain.invoke(input_dict, config=config)
        # Only print 50 chars of the answer for debugging
        print(f"Generated Answer (first 50 chars): {final_answer[:50]}...")

        # --- LLM Confidence Assessment ---
        try:
            print("Assessing generation confidence...")
            confidence_input = {"query": query, "answer": final_answer}
            confidence_assessment = confidence_chain.invoke(confidence_input, config=config)
            confidence_str = confidence_assessment.strip().lower()
            print(f"Confidence Assessment: '{confidence_str}'")

            if confidence_str == "high":
                confidence = "high"
            elif confidence_str == "low":
                confidence = "low"
            else:
                print(f"Warning: Confidence assessment returned unexpected value: '{confidence_assessment}'. Defaulting to low.")
                confidence = "low" # Default to low if output is not exactly "high" or "low"

        except Exception as conf_e:
            print(f"Error during confidence assessment: {conf_e}")
            # If confidence assessment fails, default to low confidence
            confidence = "low"
            # Optionally append this error to the main error message if needed
            generation_error = f"Confidence assessment failed: {conf_e}"

        print(f"Final Generation Confidence: {confidence}")

    except Exception as gen_e:
        print(f"Error in generate_answer_node (Generation): {gen_e}")
        final_answer = "Lo siento, ocurrió un error al generar la respuesta."
        confidence = "low"
        generation_error = f"Error generating answer: {gen_e}"

    # Return the state update
    return {
        "final_answer": final_answer,
        "generation_confidence": confidence,
        "error_message": generation_error # Pass along any error message
    }
# --- Coordinator Logic (Decision Nodes) ---

MAX_RERETRIEVAL_ATTEMPTS = 1 # Limit re-retrieval loops

# Decide after validation
def decide_after_validation(state: AgentState):
    """Decides the next step after document validation based on the validation node's output."""
    print("--- Coordinator: Deciding after validation ---")
    status = state.get("validation_status")
    attempts = state.get("reretrieval_attempts", 0) # This now reflects the count *after* the validation node potentially incremented it
    error_msg = state.get("error_message") # Check for errors from the validation node
    print(f"Validation Status: {status}, Attempts Made (including current if failed): {attempts}, Error: {error_msg}")

    # Prioritize handling errors from the validation node
    if status == "error":
        print(f"Routing to handle_error due to validation status 'error'. Message: {error_msg}")
        return "handle_error"

    # Proceed based on status if no error
    if status == "sufficient":
        print("Routing to generate_answer.")
        return "generate_answer"
    elif status == "insufficient_clarify":
        print("Routing to ask_user_for_clarification.")
        # Check if a clarification question was actually generated
        if state.get("clarification_question"):
             return "ask_user_for_clarification"
        else:
             print("Warning: Status is insufficient_clarify but no question generated by validator. Routing to error handler.")
             # Update state to reflect this specific issue
             state["error_message"] = "Validation suggested clarification, but no question was generated."
             return "handle_error" # Treat as an error state
    elif status == "insufficient_reretrieve":
        # The attempt counter was already incremented by the validation node
        if attempts <= MAX_RERETRIEVAL_ATTEMPTS:
            print(f"Routing back to retrieve_documents (Attempt {attempts} failed, next will be {attempts+1} if it happens).")
            return "retrieve_documents" # Loop back to retrieval
        else:
            print(f"Validation insufficient, max re-retrieval attempts ({MAX_RERETRIEVAL_ATTEMPTS}) reached. Ending.")
            # Set a final answer indicating failure after retries
            state["final_answer"] = "No pude encontrar información suficientemente relevante tras varios intentos."
            return END # End the graph execution
    else:
         # Catch any unexpected status values
         print(f"Unknown validation status '{status}', routing to error handler.")
         state["error_message"] = f"Unexpected validation status: {status}"
         return "handle_error"


# Decide after generation
def decide_after_generation(state: AgentState):
    """Decides the next step after answer generation."""
    print("--- Coordinator: Deciding after generation ---")
    confidence = state.get("generation_confidence")
    error_msg = state.get("error_message") # Check for errors from generation node
    print(f"Generation Confidence: {confidence}, Error: {error_msg}")

    # Prioritize handling errors from the generation node
    if error_msg:
         print(f"Routing to handle_error due to generation error: {error_msg}")
         return "handle_error"

    # Decide based on confidence if no error
    if confidence == "low":
        print("Low confidence answer. Ending.")
        return "low_confidence" 
    elif confidence == "high":
        print("High confidence answer. Ending.")
        return "high_confidence" 
    else:
        # Catch unexpected confidence values
        print(f"Unknown generation confidence '{confidence}', routing to error handler.")
        state["error_message"] = f"Unexpected generation confidence: {confidence}"
        return "handle_error"

# Simple error handling node
def handle_error_node(state: AgentState):
    """Sets a generic error message if one occurred during the process."""
    print("--- Node: Handle Error ---")
    # Use the error message already set in the state, or provide a default
    error = state.get("error_message", "Ocurrió un error desconocido durante el procesamiento.")
    print(f"Final Error State: {error}")
    # Ensure final_answer reflects the error for the user
    final_error_message = f"Lo siento, ocurrió un error: {error}"
    # Update the state directly (handle_error_node modifies state and leads to END)
    return {"final_answer": final_error_message, "error_message": error}

# Node/State to handle asking the user
def ask_user_node(state: AgentState):
    """Sets the final answer to be the clarification question generated by the validator."""
    print("--- Node: Ask User for Clarification ---")
    question = state.get("clarification_question", "Necesito más detalles. ¿Podrías reformular tu pregunta?") # Fallback question
    print(f"Setting final answer to clarification question: {question}")
    # In a real application, this node might signal the main loop to pause.
    # Here, we just set the final_answer to the question and end the graph run.
    return {"final_answer": question} # Update state and prepare to end
