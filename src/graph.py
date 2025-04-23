from langgraph.graph import StateGraph, END
from .agent import (
    AgentState,
    transform_query_node,
    retrieve_documents_node,
    validate_documents_node,
    generate_answer_node,
    handle_error_node,
    ask_user_node, # New node for clarification
    # Decision functions
    decide_after_validation,
    decide_after_generation
)

# --- Build the Multi-Agent Graph ---

print("Building the multi-agent graph...")
workflow = StateGraph(AgentState)

# Add the agent nodes
workflow.add_node("transform_query", transform_query_node)
workflow.add_node("retrieve_documents", retrieve_documents_node)
workflow.add_node("validate_documents", validate_documents_node)
workflow.add_node("generate_answer", generate_answer_node)
workflow.add_node("handle_error", handle_error_node)
workflow.add_node("ask_user_for_clarification", ask_user_node) # Add the new node

# --- Define Edges and Conditional Logic ---

# Entry point
workflow.set_entry_point("transform_query")

# Basic flow edges
workflow.add_edge("transform_query", "retrieve_documents")
workflow.add_edge("retrieve_documents", "validate_documents")

# Conditional edge after validation
workflow.add_conditional_edges(
    "validate_documents",
    decide_after_validation, # Function decides based on validation_status and attempts
    {
        "generate_answer": "generate_answer",
        "retrieve_documents": "retrieve_documents", # Loop back for re-retrieval
        "ask_user_for_clarification": "ask_user_for_clarification", # Go to ask user node
        "handle_error": "handle_error",
        END: END # Route directly to END if max attempts reached or other terminal conditions
    }
)

# Conditional edge after generation
workflow.add_conditional_edges(
    "generate_answer",
    decide_after_generation, # Function decides based on confidence
    {
        "high_confidence": END, # End if confidence is high
        "low_confidence": "ask_user_for_clarification", # Ask user for clarification if confidence is low
        "handle_error": "handle_error"
        # Add mapping for END if decide_after_generation can return it
        # END: END
    }
)

# Edges leading to termination
workflow.add_edge("handle_error", END)
workflow.add_edge("ask_user_for_clarification", END) # End after setting the clarification question as the answer


# Compile the graph
app = workflow.compile()
print("Multi-agent graph compiled.")
