import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    answer_correctness,
    answer_similarity,
)
import os
import sys
import logging
import argparse # To handle command-line arguments for batching/final eval

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Add project root ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
logging.info(f"Added project root to sys.path: {project_root}")

# --- Import RAG system components ---
try:
    from src.graph import app
    from src.config import RECURSION_LIMIT
    from langchain_core.runnables import RunnableConfig
    # Define the evaluation function (same as before)
    def run_system_for_evaluation(query: str):
        """Invokes the graph app and extracts answer/contexts."""
        inputs = {"original_query": query, "chat_history": []}
        run_config = RunnableConfig(recursion_limit=RECURSION_LIMIT)
        try:
            final_state = app.invoke(inputs, config=run_config)
            final_answer = final_state.get("final_answer", "Error: No final answer key.")
            if not isinstance(final_answer, str):
                 final_answer = str(final_answer)
            retrieved_docs = final_state.get("documents", [])
            contexts = [getattr(doc, 'page_content', str(doc)) for doc in retrieved_docs]
            if not final_answer: final_answer = "Error: Empty answer generated."
            if not contexts: logging.warning(f"No contexts extracted for query: {query}")
            return final_answer, contexts
        except Exception as e:
            logging.error(f"Error invoking graph for query '{query}': {e}", exc_info=True)
            return f"Error during graph execution: {e}", []

except ImportError as e:
    logging.error(f"Failed to import necessary modules: {e}.", exc_info=True)
    sys.exit(1)
except AttributeError as e:
     logging.error(f"Attribute error during import or setup: {e}.", exc_info=True)
     sys.exit(1)

# --- Load Ground Truth Data ---
def load_evaluation_data(csv_path: str):
    """Loads evaluation data from CSV and renames columns."""
    try:
        df = pd.read_csv(csv_path)
        df = df.rename(columns={
            "Pregunta": "question",
            "Respuesta GroundTruth": "ground_truth",
        })
        logging.info(f"Ground truth data loaded from {csv_path}. Columns: {df.columns.tolist()}")
        if "question" not in df.columns or "ground_truth" not in df.columns:
             raise ValueError("CSV must contain 'Pregunta' and 'Respuesta GroundTruth' columns.")
        df['ground_truth'] = df['ground_truth'].astype(str)
        df['question'] = df['question'].astype(str) # Ensure question is string for comparison
        return df
    except FileNotFoundError:
        logging.error(f"Error: CSV file not found at {csv_path}")
        return None
    except Exception as e:
         logging.error(f"Error loading/processing CSV: {e}", exc_info=True)
         return None

# --- Configuration ---
EVAL_CSV_PATH = os.path.join(os.path.dirname(__file__), "GroundTruth-PreguntasRespuestas.csv")
INTERMEDIATE_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "evaluation_intermediate_results.csv")
FINAL_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "ragas_evaluation_results.csv")

# --- Command Line Arguments ---
parser = argparse.ArgumentParser(description="Run RAG evaluation incrementally or perform final RAGAS evaluation.")
parser.add_argument("--start-index", type=int, default=0, help="Start processing questions from this index (0-based).")
parser.add_argument("--max-questions", type=int, default=5, help="Maximum number of *new* questions to process in this run.")
parser.add_argument("--evaluate-all", action="store_true", help="Load all intermediate results and run final RAGAS evaluation.")
args = parser.parse_args()

# --- Main Logic ---

if args.evaluate_all:
    # --- Final Evaluation Step ---
    logging.info("--- Starting Final RAGAS Evaluation ---")
    if not os.path.exists(INTERMEDIATE_RESULTS_PATH):
        logging.error(f"Intermediate results file not found: {INTERMEDIATE_RESULTS_PATH}. Cannot perform final evaluation.")
        sys.exit(1)

    try:
        results_df = pd.read_csv(INTERMEDIATE_RESULTS_PATH)
        logging.info(f"Loaded {len(results_df)} results from {INTERMEDIATE_RESULTS_PATH}")

        # Convert context strings back to lists (assuming they were saved as string representations)
        # Make sure this matches how you save them! If saving as JSON strings, use json.loads
        import ast
        try:
            results_df['contexts'] = results_df['contexts'].apply(ast.literal_eval)
        except (ValueError, SyntaxError) as e:
             logging.warning(f"Could not automatically convert 'contexts' column back to list using ast.literal_eval. Error: {e}. Ensure it's stored correctly (e.g., as a valid Python list string). Trying simple string split as fallback.")
             # Fallback or specific handling might be needed depending on how lists are saved
             results_df['contexts'] = results_df['contexts'].apply(lambda x: x.split('|||') if isinstance(x, str) else []) # Example fallback

        # Ensure correct types
        results_df['answer'] = results_df['answer'].astype(str)
        results_df['question'] = results_df['question'].astype(str)
        results_df['ground_truth'] = results_df['ground_truth'].astype(str)
        results_df['contexts'] = results_df['contexts'].apply(lambda x: list(map(str, x)) if isinstance(x, list) else [])


        ragas_dataset_dict = {
            "question": results_df["question"].tolist(),
            "answer": results_df["answer"].tolist(),
            "contexts": results_df["contexts"].tolist(),
            "ground_truth": results_df["ground_truth"].tolist()
        }
        ragas_dataset = Dataset.from_dict(ragas_dataset_dict)

        logging.info("Dataset prepared for RAGAS:")
        print(ragas_dataset)

        metrics_to_run = [
            faithfulness,
            answer_relevancy,
            context_relevancy,
            answer_correctness,
            answer_similarity
        ]

        logging.info(f"Initiating RAGAS evaluation with metrics: {[m.name for m in metrics_to_run]}")
        evaluation_result = evaluate(
            dataset=ragas_dataset,
            metrics=metrics_to_run
        )
        logging.info("RAGAS evaluation completed.")

        print("\n--- RAGAS Evaluation Results ---")
        print(evaluation_result)

        results_ragas_df = evaluation_result.to_pandas()
        print("\nRAGAS Results as DataFrame:")
        print(results_ragas_df.head())

        results_ragas_df.to_csv(FINAL_RESULTS_PATH, index=False, encoding='utf-8')
        logging.info(f"Final evaluation results saved to: {FINAL_RESULTS_PATH}")

    except Exception as e:
        logging.error(f"Error during final evaluation: {e}", exc_info=True)
        sys.exit(1)

else:
    # --- Incremental Processing Step ---
    logging.info(f"--- Starting Incremental Processing (Start Index: {args.start_index}, Max New Questions: {args.max_questions}) ---")

    eval_df = load_evaluation_data(EVAL_CSV_PATH)
    if eval_df is None:
        sys.exit(1)

    # Load existing results to avoid reprocessing
    processed_questions = set()
    if os.path.exists(INTERMEDIATE_RESULTS_PATH):
        try:
            intermediate_df = pd.read_csv(INTERMEDIATE_RESULTS_PATH)
            processed_questions = set(intermediate_df['question'].astype(str).tolist())
            logging.info(f"Loaded {len(processed_questions)} already processed questions from {INTERMEDIATE_RESULTS_PATH}")
        except Exception as e:
            logging.warning(f"Could not load or parse intermediate results file {INTERMEDIATE_RESULTS_PATH}. Starting fresh or potentially reprocessing. Error: {e}")
            processed_questions = set() # Reset if loading fails

    questions_processed_this_run = 0
    total_questions_in_gt = len(eval_df)

    for index, row in eval_df.iterrows():
        # Skip based on start index
        if index < args.start_index:
            continue

        question = row['question']
        ground_truth = row['ground_truth']

        # Skip if already processed
        if question in processed_questions:
            logging.debug(f"Skipping already processed question (Index {index}): {question[:80]}...")
            continue

        # Stop if max questions for this run reached
        if questions_processed_this_run >= args.max_questions:
            logging.info(f"Reached max number of new questions for this run ({args.max_questions}). Stopping.")
            break

        logging.info(f"Processing question {index + 1}/{total_questions_in_gt}: {question[:80]}...")
        answer, contexts = run_system_for_evaluation(question)

        # Prepare data for saving - IMPORTANT: Save contexts as a string representation
        # Using a simple delimiter unlikely to be in the text, or use json.dumps
        contexts_str = "|||".join(contexts) # Simple delimiter example
        # contexts_str = json.dumps(contexts) # Safer JSON approach

        result_data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts_str], # Save string representation
            "ground_truth": [ground_truth]
        }
        result_df_single = pd.DataFrame(result_data)

        # Append result to the intermediate CSV
        try:
            file_exists = os.path.exists(INTERMEDIATE_RESULTS_PATH)
            result_df_single.to_csv(
                INTERMEDIATE_RESULTS_PATH,
                mode='a',          # Append mode
                header=not file_exists, # Write header only if file doesn't exist
                index=False,
                encoding='utf-8'
            )
            logging.info(f"Saved result for question index {index} to {INTERMEDIATE_RESULTS_PATH}")
            processed_questions.add(question) # Add to processed set for this run's logic
            questions_processed_this_run += 1
        except Exception as e:
            logging.error(f"Error saving intermediate result for question index {index}: {e}", exc_info=True)
            # Decide if you want to stop or continue on save error

    logging.info(f"Incremental processing run finished. Processed {questions_processed_this_run} new questions.")
    remaining_questions = total_questions_in_gt - len(processed_questions)
    logging.info(f"Total processed questions: {len(processed_questions)}. Estimated remaining: {remaining_questions}")
    if remaining_questions > 0:
         next_start_index = args.start_index + questions_processed_this_run # Rough estimate
         logging.info(f"To continue, run again potentially starting around index {next_start_index} or check the intermediate file.")
    else:
         logging.info("All questions seem to be processed. Run with --evaluate-all to get final RAGAS scores.")


print("\n--- Fin de la Evaluación ---")