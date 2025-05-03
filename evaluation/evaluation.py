import pandas as pd
import time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness, # 
    answer_relevancy,
    context_precision, # Measures ranking/proportion of relevant context
    context_recall,    # Measures if all necessary info was retrieved
    ContextRelevance, # Measures relevance of individual retrieved chunks
    answer_correctness,# Compares answer to ground truth
    answer_similarity  # Compares answer to ground truth semantically
)
import os
import sys
import logging
import argparse # To handle command-line arguments for batching/final eval
import ast # To parse string representation of list back to list
import json # Import json
import csv # Import csv for quoting constants

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
            # Use "validated_docs" which contains the documents used for generation
            retrieved_docs = final_state.get("validated_docs", []) 
            # Ensure contexts are strings
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
        # Be more explicit with parsing options for quoted fields and spaces
        df = pd.read_csv(
            csv_path,
            on_bad_lines='warn',
            engine='python',
            quotechar='"',             # Explicitly define the quote character
            skipinitialspace=True      # Handle potential spaces after commas
        )

        # Clean up column names (remove leading/trailing spaces and quotes)
        df.columns = df.columns.str.strip().str.strip('"').str.strip()

        # Rename columns *after* cleaning
        df = df.rename(columns={
            "Pregunta": "question",
            "Respuesta GroundTruth": "ground_truth",
        })
        logging.info(f"Ground truth data loaded from {csv_path}. Cleaned Columns: {df.columns.tolist()}")

        # Check required columns again after cleaning and renaming
        if "question" not in df.columns or "ground_truth" not in df.columns:
             raise ValueError("CSV must contain 'Pregunta' and 'Respuesta GroundTruth' columns after cleaning.")

        df['ground_truth'] = df['ground_truth'].astype(str)
        df['question'] = df['question'].astype(str) # Ensure question is string for comparison

        # Reset index to ensure it's a simple integer index
        df = df.reset_index(drop=True)
        logging.info("DataFrame index reset.")

        return df
    except FileNotFoundError:
        logging.error(f"Error: CSV file not found at {csv_path}")
        return None
    except Exception as e:
         # Log the specific pandas parsing error if it occurs
         if isinstance(e, pd.errors.ParserError):
             logging.error(f"Error parsing CSV file {csv_path}: {e}", exc_info=False) # Keep log cleaner
             logging.error("Check the CSV file for formatting issues, especially around the line number mentioned in the error (if any). Ensure commas within fields are properly quoted.")
         else:
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


# --- Instantiate Models (Example) ---
# Ensure relevant API keys (e.g., OPENAI_API_KEY) are set in your environment
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from src import config
try:
   
    judge_llm = AzureChatOpenAI(
        deployment_name="gpt-4o-mini",
        model_name="gpt-4o-mini",
        api_version="2024-12-01-preview",
        azure_endpoint=config.JUDGE_ENDPOINT,
        api_key=config.JUDGE_KEY,
        max_retries=5,
)
    
    # RAGAS often uses embeddings for similarity checks
    embedding_model = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-large-2",
        model="text-embedding-3-large",
        azure_endpoint=config.EMBEDDING_EVAL_ENDPOINT,
        api_key=config.EMBEDDING_EVAL_KEY,
        api_version="2024-02-01",
        max_retries=5,
        chunk_size=1, # Set to 1 for better performance with large texts
    )

except Exception as e:
    logging.error(f"Failed to instantiate LLM or Embeddings. Check API keys and packages: {e}")
    sys.exit(1)

# --- Main Logic ---

if args.evaluate_all:
    # --- Final Evaluation Step ---
    logging.info("--- Starting Final RAGAS Evaluation ---")
    if not os.path.exists(INTERMEDIATE_RESULTS_PATH):
        logging.error(f"Intermediate results file not found: {INTERMEDIATE_RESULTS_PATH}. Cannot perform final evaluation.")
        sys.exit(1)

    try:
        # Read intermediate results robustly
        results_df = pd.read_csv(
            INTERMEDIATE_RESULTS_PATH,
            quotechar='"',
            engine='python',
            quoting=csv.QUOTE_NONNUMERIC,
            doublequote=True
        )
        logging.info(f"Loaded {len(results_df)} results from {INTERMEDIATE_RESULTS_PATH}")

        # --- CHUNKING LOGIC START ---
        chunk_size = 3 # Adjust as needed based on rate limits
        all_results_list = []
        total_rows = len(results_df)
        logging.info(f"Processing {total_rows} results in chunks of {chunk_size}")

        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            logging.info(f"--- Evaluating chunk: rows {start} to {end-1} ---")
            chunk_df = results_df.iloc[start:end].copy() # Use .copy() to avoid SettingWithCopyWarning

            # Convert context JSON strings back to lists for the chunk
            chunk_df['contexts'] = chunk_df['contexts'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

            # Ensure correct types for the chunk
            chunk_df['answer'] = chunk_df['answer'].astype(str)
            chunk_df['question'] = chunk_df['question'].astype(str)
            chunk_df['ground_truth'] = chunk_df['ground_truth'].astype(str)

            ragas_dataset_dict = {
                "question": chunk_df["question"].tolist(),
                "answer": chunk_df["answer"].tolist(),
                "contexts": chunk_df["contexts"].tolist(),
                "ground_truth": chunk_df["ground_truth"].tolist()
            }
            ragas_dataset = Dataset.from_dict(ragas_dataset_dict)

            logging.info(f"Dataset chunk prepared for RAGAS (Size: {len(ragas_dataset)}):")

            # Define metrics
            context_relevance_metric = ContextRelevance()
            metrics_to_run = [
                faithfulness, answer_relevancy, context_precision, context_recall,
                context_relevance_metric, answer_correctness, answer_similarity
            ]
            logging.info(f"Initiating RAGAS evaluation for chunk with metrics: {[m.name for m in metrics_to_run]}")

            try:
                evaluation_result = evaluate(
                    dataset=ragas_dataset,
                    metrics=metrics_to_run,
                    llm=judge_llm,
                    embeddings=embedding_model,
                    raise_exceptions=False # Log errors per row instead of stopping the whole chunk
                )
                logging.info(f"RAGAS evaluation completed for chunk {start}-{end-1}.")
                chunk_results_df = evaluation_result.to_pandas()
                all_results_list.append(chunk_results_df)

            except Exception as chunk_e:
                logging.error(f"Error evaluating chunk {start}-{end-1}: {chunk_e}", exc_info=True)
                # Optionally add logic here to save partial results or stop

            # Add a delay between chunks
            delay_seconds = 30 # Adjust as needed
            logging.info(f"Waiting {delay_seconds} seconds before next chunk...")
            time.sleep(delay_seconds)
        # --- CHUNKING LOGIC END ---

        if all_results_list:
            # Combine results from all chunks
            final_ragas_df = pd.concat(all_results_list, ignore_index=True)

            print("\n--- Combined RAGAS Evaluation Results ---")
            print(final_ragas_df.head())

            final_ragas_df.to_csv(FINAL_RESULTS_PATH, index=False, encoding='utf-8')
            logging.info(f"Final evaluation results saved to: {FINAL_RESULTS_PATH}")
        else:
            logging.warning("No results were generated from any chunk.")

    except Exception as e:
        logging.error(f"Error during final evaluation setup or file loading: {e}", exc_info=True)
        sys.exit(1)

else:
    # --- Incremental Processing Step ---
    logging.info(f"--- Starting Incremental Processing (Start Index: {args.start_index}, Max New Questions: {args.max_questions}) ---")

    eval_df = load_evaluation_data(EVAL_CSV_PATH)
    if eval_df is None:
        sys.exit(1)

    # Load existing results to avoid reprocessing
    processed_questions = set()
    intermediate_results_list = [] # Store existing results to rewrite file later if needed
    if os.path.exists(INTERMEDIATE_RESULTS_PATH):
        try:
            # Use robust reading here too, ensuring doublequote handling
            intermediate_df = pd.read_csv(
                INTERMEDIATE_RESULTS_PATH,
                quotechar='"',
                engine='python',
                quoting=csv.QUOTE_NONNUMERIC, # Match writing strategy
                doublequote=True              # Handle doubled quotes during reading
                )
            # Ensure question column is string for comparison
            intermediate_df['question'] = intermediate_df['question'].astype(str)
            processed_questions = set(intermediate_df['question'].tolist())
            logging.info(f"Loaded {len(processed_questions)} already processed questions from {INTERMEDIATE_RESULTS_PATH}")
        except Exception as e:
            logging.warning(f"Could not load or parse intermediate results file {INTERMEDIATE_RESULTS_PATH}. Starting fresh or potentially reprocessing. Error: {e}")
            processed_questions = set()

    questions_processed_this_run = 0
    total_questions_in_gt = len(eval_df)
    new_results_list = [] # Collect new results for this run

    for index, row in eval_df.iterrows():
        # Skip based on start index
        if index < args.start_index:
            continue

        question = row['question']
        ground_truth = row['ground_truth']

        # Skip if already processed (check as string)
        if str(question) in processed_questions:
            logging.debug(f"Skipping already processed question (Index {index}): {str(question)[:80]}...")
            continue

        # Stop if max questions for this run reached
        if questions_processed_this_run >= args.max_questions:
            logging.info(f"Reached max number of new questions for this run ({args.max_questions}). Stopping.")
            break

        logging.info(f"Processing question {index + 1}/{total_questions_in_gt}: {str(question)[:80]}...")
        answer, contexts = run_system_for_evaluation(question)

        # Prepare data for saving - Save contexts as JSON string
        try:
            # Use json.dumps for robust serialization, ensure_ascii=False for non-latin chars
            contexts_str = json.dumps(contexts, ensure_ascii=False)
        except TypeError as e:
            logging.error(f"Could not serialize contexts to JSON for question '{question}': {e}. Saving as empty list string.")
            contexts_str = "[]"


        result_data = {
            "question": question,
            "answer": answer,
            "contexts": contexts_str, # Save JSON string representation
            "ground_truth": ground_truth
        }
        new_results_list.append(result_data)
        processed_questions.add(str(question)) # Add to processed set for this run's logic
        questions_processed_this_run += 1

    # Append new results to the intermediate CSV with proper quoting and escaping
    if new_results_list:
        new_results_df = pd.DataFrame(new_results_list)
        try:
            file_exists = os.path.exists(INTERMEDIATE_RESULTS_PATH) and os.path.getsize(INTERMEDIATE_RESULTS_PATH) > 0
            new_results_df.to_csv(
                INTERMEDIATE_RESULTS_PATH,
                mode='a',          # Append mode
                header=not file_exists, # Write header only if file doesn't exist or is empty
                index=False,
                encoding='utf-8',
                quoting=csv.QUOTE_NONNUMERIC, # Quote fields containing delimiters, quotes, or newlines
                doublequote=True              # Ensure internal quotes are doubled
            )
            logging.info(f"Appended {len(new_results_list)} new results to {INTERMEDIATE_RESULTS_PATH}")
        except Exception as e:
            logging.error(f"Error saving intermediate results: {e}", exc_info=True)

    logging.info(f"Incremental processing run finished. Processed {questions_processed_this_run} new questions.")
    current_total_processed = 0
    if os.path.exists(INTERMEDIATE_RESULTS_PATH):
         try:
             current_total_processed = len(pd.read_csv(INTERMEDIATE_RESULTS_PATH))
         except Exception:
             current_total_processed = len(processed_questions)

    remaining_questions = total_questions_in_gt - current_total_processed
    logging.info(f"Total processed questions now: {current_total_processed}. Estimated remaining: {max(0, remaining_questions)}")

    if remaining_questions > 0:
         next_start_index = args.start_index + questions_processed_this_run
         logging.info(f"To continue, run again potentially starting around index {next_start_index} or check the intermediate file.")
    else:
         logging.info("All questions seem to be processed based on count. Run with --evaluate-all to get final RAGAS scores.")


print("\n--- Fin de la Evaluación ---")