import pandas as pd
import time
import os
import sys
import logging
import argparse  # To handle command-line arguments for batching/final eval
import ast  # To parse string representation of list back to list
import json  # Import json
import csv  # Import csv for quoting constants
from importlib import metadata as importlib_metadata
import subprocess
import asyncio
from typing import Any, List
from urllib import request as urllib_request
from urllib import error as urllib_error

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _pkg_version(package_name: str) -> str:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"
    except Exception:
        return "unknown"

def _module_origin(module_name: str) -> str:
    try:
        module = __import__(module_name)
        module_file = getattr(module, "__file__", None)
        module_spec = getattr(module, "__spec__", None)
        if module_file:
            return module_file
        if module_spec and getattr(module_spec, "origin", None):
            return module_spec.origin
        return "unknown"
    except Exception:
        return "unknown"

def _ensure_mistral_symbol():
    """Patch mistralai so instructor/ragas can import Mistral consistently."""
    try:
        import mistralai
    except Exception as e:
        raise ImportError("mistralai could not be imported.") from e

    if hasattr(mistralai, "Mistral"):
        return

    candidate_locations = (
        ("mistralai.client", "Mistral"),
        ("mistralai.client", "MistralClient"),
        ("mistralai", "MistralClient"),
    )

    for module_name, symbol_name in candidate_locations:
        try:
            candidate_module = __import__(module_name, fromlist=[symbol_name])
            symbol = getattr(candidate_module, symbol_name, None)
            if symbol is not None:
                setattr(mistralai, "Mistral", symbol)
                logging.warning(
                    "Patched mistralai.Mistral from %s.%s for instructor/ragas compatibility.",
                    module_name,
                    symbol_name,
                )
                return
        except Exception:
            continue

    raise ImportError("mistralai does not expose a compatible Mistral symbol.")

def _validate_mistralai_compatibility():
    """Fail fast with a clear message when mistralai/instructor are mismatched."""
    try:
        import mistralai  # noqa: F401
        _ensure_mistral_symbol()
        from mistralai import Mistral  # noqa: F401
        return
    except Exception as e:
        logging.error("Detected incompatible mistralai installation for ragas/instructor.")
        logging.error(f"mistralai version: {_pkg_version('mistralai')}")
        logging.error(f"instructor version: {_pkg_version('instructor')}")
        logging.error(f"ragas version: {_pkg_version('ragas')}")
        logging.error(f"mistralai origin: {_module_origin('mistralai')}")
        logging.error(f"Import failure: {e}", exc_info=True)
        logging.error(
            "Suggested fix: reinstall compatible versions or remove the broken mistralai package."
        )
        logging.error(
            "Example:\n"
            "  pip uninstall -y mistralai instructor ragas\n"
            "  pip install -U mistralai instructor ragas"
        )
        raise ImportError(
            "mistralai is incompatible with instructor/ragas in this environment."
        ) from e

def _safe_json_loads(value):
    """Parse JSON if possible; otherwise return the original value."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value

class AwaitableList(list):
    """List that can be awaited (returns itself) for async compatibility."""
    def __await__(self):
        async def _wrap():
            return self
        return _wrap().__await__()

def _normalize_prompt_value(value: Any) -> str:
    """Coerce various prompt objects into a plain string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8", errors="ignore")
        except Exception:
            return str(value)
    if isinstance(value, dict):
        if "content" in value:
            return str(value["content"])
        if "text" in value:
            return str(value["text"])
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    to_string = getattr(value, "to_string", None)
    if callable(to_string):
        try:
            return to_string()
        except Exception:
            pass
    if hasattr(value, "content"):
        try:
            return str(value.content)
        except Exception:
            pass
    if hasattr(value, "text"):
        try:
            return str(value.text)
        except Exception:
            pass
    if hasattr(value, "messages"):
        try:
            messages = value.messages
            return "\n".join(_normalize_prompt_value(m) for m in messages)
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        # Treat tuples that look like (prompt, stop/kwargs) as a single prompt
        if isinstance(value, tuple) and len(value) == 2:
            try:
                return _normalize_prompt_value(value[0])
            except Exception:
                pass
        return "\n".join(_normalize_prompt_value(v) for v in value)
    return str(value)

def _normalize_contexts(value: Any) -> List[str]:
    """Ensure contexts are always a list of strings."""
    if value is None:
        return []
    try:
        if isinstance(value, float) and pd.isna(value):
            return []
    except Exception:
        pass

    if isinstance(value, list):
        return [str(_normalize_prompt_value(v)) for v in value if v is not None]
    if isinstance(value, tuple):
        return [str(_normalize_prompt_value(v)) for v in value if v is not None]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            return _normalize_contexts(parsed)
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(s)
            return _normalize_contexts(parsed)
        except Exception:
            pass
        return [s]
    return [str(_normalize_prompt_value(value))]

# --- Load Intermediate Results ---
def _load_intermediate_results(csv_path: str) -> pd.DataFrame:
    """Load intermediate results robustly, recovering malformed rows when possible."""
    expected_columns = ["question", "answer", "contexts", "ground_truth"]

    try:
        return pd.read_csv(
            csv_path,
            quotechar='"',
            engine='python',
            quoting=csv.QUOTE_NONNUMERIC,
            doublequote=True,
        )
    except pd.errors.ParserError as e:
        logging.warning(
            "Standard CSV parsing failed for %s: %s. Falling back to tolerant parsing.",
            csv_path,
            e,
        )

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(
            f,
            quotechar='"',
            doublequote=True,
            skipinitialspace=True,
        )
        headers = next(reader, None)
        if not headers:
            return pd.DataFrame(columns=expected_columns)

        headers = [h.strip().strip('"').strip() for h in headers]
        if len(headers) < 4:
            headers = expected_columns

        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue

            if len(row) == 4:
                rows.append(row)
                continue

            if len(row) < 4:
                logging.warning(
                    "Skipping malformed row %s in %s: expected at least 4 fields, saw %s",
                    line_no,
                    csv_path,
                    len(row),
                )
                continue

            recovered = [row[0], row[1], ",".join(row[2:-1]), row[-1]]
            logging.warning(
                "Recovered malformed row %s in %s by joining extra fields into contexts.",
                line_no,
                csv_path,
            )
            rows.append(recovered)

    return pd.DataFrame(rows, columns=headers[:4])

# --- Lazy import helper for RAGAS-only dependencies ---
def load_ragas_dependencies():
    """Import RAGAS stack only when final evaluation is requested."""
    try:
        _validate_mistralai_compatibility()

        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        try:
            from ragas.metrics.collections import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                ContextRelevance,
                answer_correctness,
                answer_similarity,
            )
        except ImportError:
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                ContextRelevance,
                answer_correctness,
                answer_similarity,
            )

        return {
            "Dataset": Dataset,
            "evaluate": ragas_evaluate,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "ContextRelevance": ContextRelevance,
            "answer_correctness": answer_correctness,
            "answer_similarity": answer_similarity,
        }
    except ImportError as e:
        logging.error(f"Failed to import RAGAS dependencies: {e}", exc_info=True)
        logging.error(
            "Final evaluation requires compatible versions of ragas/instructor/mistralai. "
            "If running incremental generation only, omit --evaluate-all."
        )
        logging.error(
            "Suggested fix: pip uninstall -y mistralai instructor ragas && pip install -U mistralai instructor ragas"
        )
        raise

# --- Add project root ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
logging.info(f"Added project root to sys.path: {project_root}")

# --- Lazy import for RAG runtime components ---
_RAG_RUNTIME_CACHE = None

def load_rag_runtime_dependencies():
    """Import graph runtime only when incremental generation actually needs it."""
    global _RAG_RUNTIME_CACHE
    if _RAG_RUNTIME_CACHE is not None:
        return _RAG_RUNTIME_CACHE

    try:
        from src.graph import app
        from src.config import RECURSION_LIMIT
        from langchain_core.runnables import RunnableConfig
        _RAG_RUNTIME_CACHE = (app, RECURSION_LIMIT, RunnableConfig)
        return _RAG_RUNTIME_CACHE
    except ImportError as e:
        logging.error(f"Failed to import RAG runtime dependencies: {e}", exc_info=True)
        if "_ON_EMIT_RECURSION_COUNT_KEY" in str(e):
            logging.error("Detected OpenTelemetry version mismatch in dependency chain (chromadb/langchain_chroma).")
            logging.error("Suggested fix:")
            logging.error("  pip install -U opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc")
            logging.error("  pip install -U chromadb langchain-chroma")
        raise
    except AttributeError as e:
        logging.error(f"Attribute error during RAG runtime import: {e}", exc_info=True)
        raise

# Define the evaluation function (same as before, now with lazy runtime import)
def run_system_for_evaluation(query: str):
    """Invokes the graph app and extracts answer/contexts.

    Adds a conservative fallback if no supporting contexts are retrieved: the agent
    should avoid definitive medical recommendations when evidence is lacking.
    """
    try:
        app, RECURSION_LIMIT, RunnableConfig = load_rag_runtime_dependencies()
    except Exception as e:
        return f"Error importing RAG runtime: {e}", []

    inputs = {"original_query": query, "chat_history": []}
    run_config = RunnableConfig(recursion_limit=RECURSION_LIMIT)
    try:
        final_state = app.invoke(inputs, config=run_config)
        final_answer = final_state.get("final_answer", "Error: No final answer key.")
        if not isinstance(final_answer, str):
            final_answer = str(final_answer)
        retrieved_docs = final_state.get("validated_docs", [])
        contexts = [getattr(doc, 'page_content', str(doc)) for doc in retrieved_docs]

        if not contexts:
            conservative = "I don't have enough reliable information to answer that question confidently. Please consult a healthcare professional."
            logging.warning(f"No contexts extracted for query: {query} — returning conservative fallback.")
            return conservative, []

        if not final_answer:
            final_answer = "Error: Empty answer generated."

        return final_answer, contexts
    except Exception as e:
        logging.error(f"Error invoking graph for query '{query}': {e}", exc_info=True)
        return f"Error during graph execution: {e}", []

# --- Load Ground Truth Data ---
def load_evaluation_data(csv_path: str):
    """Loads evaluation data from CSV and renames columns."""
    try:
        df = pd.read_csv(
            csv_path,
            on_bad_lines='warn',
            engine='python',
            quotechar='"',
            skipinitialspace=True
        )

        df.columns = df.columns.str.strip().str.strip('"').str.strip()

        df = df.rename(columns={
            "Pregunta": "question",
            "Respuesta GroundTruth": "ground_truth",
        })
        logging.info(f"Ground truth data loaded from {csv_path}. Cleaned Columns: {df.columns.tolist()}")

        if "question" not in df.columns or "ground_truth" not in df.columns:
            raise ValueError("CSV must contain 'Pregunta' and 'Respuesta GroundTruth' columns after cleaning.")

        df['ground_truth'] = df['ground_truth'].astype(str)
        df['question'] = df['question'].astype(str)
        df = df.reset_index(drop=True)
        logging.info("DataFrame index reset.")

        return df
    except FileNotFoundError:
        logging.error(f"Error: CSV file not found at {csv_path}")
        return None
    except Exception as e:
        if isinstance(e, pd.errors.ParserError):
            logging.error(f"Error parsing CSV file {csv_path}: {e}", exc_info=False)
            logging.error("Check the CSV file for formatting issues, especially around the line number mentioned in the error (if any). Ensure commas within fields are properly quoted.")
        else:
            logging.error(f"Error loading/processing CSV: {e}", exc_info=True)
        return None

# --- Configuration ---
EVAL_CSV_PATH = os.path.join(os.path.dirname(__file__), "GroundTruth-PreguntasRespuestas.csv")
INTERMEDIATE_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "evaluation_intermediate_results.csv")
FINAL_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "ragas_evaluation_results.csv")

# Insert Ollama CLI fallback wrapper so it's always defined when used below.
class OllamaCLIWrapper:
    """Fallback wrapper that uses Ollama's HTTP API first, then CLI as a last resort."""
    def __init__(self, model: str = "mistral", default_timeout: int | None = None):
        self.model = model
        self.base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        env_timeout = os.getenv("OLLAMA_CLI_TIMEOUT")
        if default_timeout is None and env_timeout:
            try:
                default_timeout = int(env_timeout)
            except ValueError:
                default_timeout = None
        if default_timeout is None:
            default_timeout = 300
        self.default_timeout = default_timeout

    def _coerce_timeout(self, value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _resolve_timeout(self, timeout: int | None, kwargs: dict) -> int:
        if timeout is None:
            for key in ("request_timeout", "timeout_s", "request_timeout_s", "timeout_seconds"):
                if key in kwargs:
                    timeout = kwargs.pop(key)
                    break
        resolved = self._coerce_timeout(timeout)
        if resolved is None or resolved <= 0:
            resolved = self.default_timeout
        return resolved

    def _call_http_api(self, endpoint: str, payload: dict, timeout: int) -> str:
        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            for key in ("response", "message", "content", "text"):
                if key in parsed and parsed[key]:
                    return str(parsed[key])
        return raw.strip()

    def _generate_http(self, prompt_text: str, timeout: int) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt_text,
            "stream": False,
            "keep_alive": "30m",
        }
        try:
            return self._call_http_api("/api/generate", payload, timeout)
        except (urllib_error.URLError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError):
            pass

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt_text}],
            "stream": False,
            "keep_alive": "30m",
        }
        return self._call_http_api("/api/chat", payload, timeout)

    def _run_cmd(self, cmd: list[str], timeout: int) -> tuple[str | None, str]:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as te:
            raise RuntimeError(f"ollama CLI timed out after {timeout}s") from te

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()

        if proc.returncode == 0:
            return stdout, ""
        return None, stderr

    def __call__(self, prompt: Any, timeout: int | None = None, **kwargs) -> str:
        timeout = self._resolve_timeout(timeout, kwargs)
        prompt_text = _normalize_prompt_value(prompt)

        # Fast path: local Ollama HTTP API.
        try:
            return self._generate_http(prompt_text, timeout)
        except Exception as http_err:
            logging.warning("Ollama HTTP API failed, falling back to CLI: %s", http_err)

        # Last resort: CLI.
        candidates = [
            ["ollama", "run", str(self.model), prompt_text],
            ["ollama", "run", str(self.model), "--prompt", prompt_text],
            ["ollama", "generate", str(self.model), "--prompt", prompt_text],
        ]

        last_err = ""
        for cmd in candidates:
            out, err = self._run_cmd(cmd, timeout)
            if out is not None:
                return out
            last_err = err

        raise RuntimeError(f"ollama CLI error: {last_err}")

    # Compatibility adapters expected by some LLM wrappers (ragas/langchain)
    def generate(self, prompts: Any, **kwargs) -> Any:
        if isinstance(prompts, list):
            prompt_list = prompts
        else:
            prompt_list = [prompts]

        class Gen:
            def __init__(self, text: str):
                self.text = text

        class Result:
            def __init__(self, gens: List[Gen]):
                self.generations = [[g] for g in gens]

        gens = []
        for p in prompt_list:
            txt = self(p, **kwargs)
            gens.append(Gen(txt))
        return Result(gens)

    async def agenerate(self, prompts: Any, **kwargs) -> Any:
        return await asyncio.to_thread(self.generate, prompts, **kwargs)

    def generate_text(self, prompt: Any, **kwargs) -> str:
        return self(prompt, **kwargs)

    async def agenerate_text(self, prompt: Any, **kwargs) -> str:
        return await asyncio.to_thread(self.generate_text, prompt, **kwargs)

    def predict(self, prompt: Any, **kwargs) -> str:
        return self(prompt, **kwargs)

    def invoke(self, input: Any, **kwargs) -> str:
        return self(input, **kwargs)

if __name__ == "__main__":
    # --- Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Run RAG evaluation incrementally or perform final RAGAS evaluation.")
    parser.add_argument("--start-index", type=int, default=0, help="Start processing questions from this index (0-based).")
    parser.add_argument("--max-questions", type=int, default=5, help="Maximum number of *new* questions to process in this run.")
    parser.add_argument("--evaluate-all", action="store_true", help="Load all intermediate results and run final RAGAS evaluation.")
    args = parser.parse_args()

    # --- Main Logic ---
    if args.evaluate_all:
        # Lazy-import local Ollama LLM + local embeddings with resilient fallbacks.
        # 1) Prefer LangChain's Ollama if available.
        # 2) Else try the Python 'ollama' client and wrap it.
        # 3) If neither is available, FALL BACK to the CLI wrapper defined above (OllamaCLIWrapper).
        try:
            from langchain.llms import Ollama as LangChainOllama  # type: ignore
            OllamaClass = LangChainOllama
            logging.info("Using LangChain's Ollama LLM class.")
        except Exception as e_lang:
            logging.info("LangChain Ollama not available: %s", e_lang)
            try:
                from ollama import Ollama as RawOllama  # type: ignore

                class OllamaWrapper:
                    """Minimal wrapper around the raw ollama client providing a callable interface."""
                    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
                        try:
                            self.client = RawOllama(base_url=base_url)
                        except TypeError:
                            self.client = RawOllama()
                        self.model = model

                    def __call__(self, prompt: str, **kwargs) -> str:
                        for method_name in ("generate", "predict", "complete", "completion", "chat"):
                            fn = getattr(self.client, method_name, None)
                            if fn is None:
                                continue
                            try:
                                try:
                                    resp = fn(prompt, model=self.model, **kwargs)
                                except TypeError:
                                    resp = fn(prompt, **kwargs)
                                if isinstance(resp, dict):
                                    for k in ("text", "result", "content"):
                                        if k in resp:
                                            return resp[k]
                                    return str(resp)
                                text = getattr(resp, "text", None)
                                if isinstance(text, str):
                                    return text
                                if isinstance(resp, str):
                                    return resp
                                return str(resp)
                            except Exception:
                                continue
                        raise RuntimeError("Ollama client has no usable generate/predict method.")

                OllamaClass = OllamaWrapper
                logging.info("Using raw 'ollama' python client with OllamaWrapper.")
            except Exception as e_ollama:
                logging.warning("Python 'ollama' client not usable: %s", e_ollama)
                logging.info("Falling back to OllamaCLIWrapper (uses the 'ollama' CLI).")
                OllamaClass = OllamaCLIWrapper

        # Embeddings: try LangChain's HuggingFaceEmbeddings, else fallback to sentence-transformers wrapper.
        try:
            from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
            EmbeddingsClass = HuggingFaceEmbeddings
            use_langchain_embeddings = True
            logging.info("Using LangChain's HuggingFaceEmbeddings.")
        except Exception:
            use_langchain_embeddings = False
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np

                class SentenceTransformersWrapper:
                    """Provide embed_documents/embed_query compatible with langchain-style embeddings."""
                    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
                        self.model = SentenceTransformer(model_name)

                    def embed_documents(self, texts):
                        texts = [_normalize_prompt_value(t) for t in texts]
                        vectors = self.model.encode(texts, show_progress_bar=False)
                        return AwaitableList([list(map(float, vec)) for vec in vectors])

                    def embed_query(self, text):
                        text = _normalize_prompt_value(text)
                        v = self.model.encode([text], show_progress_bar=False)[0]
                        return AwaitableList(list(map(float, v)))

                    def embed_text(self, text: str) -> List[float]:
                        return self.embed_query(text)

                    async def aembed_documents(self, texts):
                        return await asyncio.to_thread(self.embed_documents, texts)

                    async def aembed_query(self, text):
                        return await asyncio.to_thread(self.embed_query, text)

                    async def aembed_text(self, text: str):
                        return await asyncio.to_thread(self.embed_text, text)

                EmbeddingsClass = SentenceTransformersWrapper
                logging.info("Using sentence-transformers via SentenceTransformersWrapper for embeddings.")
            except Exception as e:
                logging.error("Failed to import any local embeddings backend: %s", e, exc_info=True)
                logging.error("Install 'sentence-transformers' or 'langchain' with HuggingFaceEmbeddings support.")
                sys.exit(1)

        # Instantiate judge_llm and embedding_model
        try:
            # Try constructing with (model, base_url) first (works for LangChain/RPC wrappers)
            judge_llm = OllamaClass(model="mistral", base_url="http://localhost:11434")
        except TypeError:
            # Fallback: construct with only model (works for CLI wrapper and simple wrappers)
            judge_llm = OllamaClass("mistral")

        try:
            if use_langchain_embeddings:
                embedding_model = EmbeddingsClass(model_name="all-MiniLM-L6-v2")
            else:
                embedding_model = EmbeddingsClass("all-MiniLM-L6-v2")
        except Exception as e:
            logging.error("Failed to instantiate embedding model: %s", e, exc_info=True)
            sys.exit(1)

        logging.info("--- Starting Final RAGAS Evaluation ---")

        try:
            ragas_deps = load_ragas_dependencies()
            Dataset = ragas_deps["Dataset"]
            ragas_evaluate = ragas_deps["evaluate"]
            faithfulness = ragas_deps["faithfulness"]
            answer_relevancy = ragas_deps["answer_relevancy"]
            context_precision = ragas_deps["context_precision"]
            context_recall = ragas_deps["context_recall"]
            ContextRelevance = ragas_deps["ContextRelevance"]
            answer_correctness = ragas_deps["answer_correctness"]
            answer_similarity = ragas_deps["answer_similarity"]
        except ImportError:
            sys.exit(1)

        if not os.path.exists(INTERMEDIATE_RESULTS_PATH):
            logging.error(f"Intermediate results file not found: {INTERMEDIATE_RESULTS_PATH}. Cannot perform final evaluation.")
            sys.exit(1)

        try:
            results_df = _load_intermediate_results(INTERMEDIATE_RESULTS_PATH)
            if results_df.empty:
                logging.error("No usable rows could be loaded from %s. Cannot perform final evaluation.", INTERMEDIATE_RESULTS_PATH)
                sys.exit(1)

            logging.info(f"Loaded {len(results_df)} results from {INTERMEDIATE_RESULTS_PATH}")

            chunk_size = 5
            all_results_list = []
            total_rows = len(results_df)
            logging.info(f"Processing {total_rows} results in chunks of {chunk_size}")

            for start in range(0, total_rows, chunk_size):
                end = min(start + chunk_size, total_rows)
                logging.info(f"--- Evaluating chunk: rows {start} to {end-1} ---")
                chunk_df = results_df.iloc[start:end].copy()

                chunk_df["contexts"] = chunk_df["contexts"].apply(_safe_json_loads)
                chunk_df["contexts"] = chunk_df["contexts"].apply(_normalize_contexts)
                chunk_df["answer"] = chunk_df["answer"].astype(str)
                chunk_df["question"] = chunk_df["question"].astype(str)
                chunk_df["ground_truth"] = chunk_df["ground_truth"].astype(str)

                ragas_dataset_dict = {
                    "question": chunk_df["question"].tolist(),
                    "answer": chunk_df["answer"].tolist(),
                    "contexts": chunk_df["contexts"].tolist(),
                    "ground_truth": chunk_df["ground_truth"].tolist()
                }
                ragas_dataset = Dataset.from_dict(ragas_dataset_dict)

                logging.info(f"Dataset chunk prepared for RAGAS (Size: {len(ragas_dataset)}):")

                context_relevance_metric = ContextRelevance()
                metrics_to_run = [
                    faithfulness, answer_relevancy, context_precision, context_recall,
                    context_relevance_metric, answer_correctness, answer_similarity
                ]
                logging.info(f"Initiating RAGAS evaluation for chunk with metrics: {[m.name for m in metrics_to_run]}")

                try:
                    evaluation_result = ragas_evaluate(
                        dataset=ragas_dataset,
                        metrics=metrics_to_run,
                        llm=judge_llm,
                        embeddings=embedding_model,
                        raise_exceptions=False
                    )
                    logging.info(f"RAGAS evaluation completed for chunk {start}-{end-1}.")
                    chunk_results_df = evaluation_result.to_pandas()
                    all_results_list.append(chunk_results_df)

                except Exception as chunk_e:
                    logging.error(f"Error evaluating chunk {start}-{end-1}: {chunk_e}", exc_info=True)

                delay_seconds = int(os.getenv("EVAL_CHUNK_DELAY_SECONDS", "0"))
                if delay_seconds > 0:
                    logging.info(f"Waiting {delay_seconds} seconds before next chunk...")
                    time.sleep(delay_seconds)

            if all_results_list:
                final_ragas_df = pd.concat(all_results_list, ignore_index=True)

                print("\n--- Combined RAGAS Evaluation Results ---")
                print(final_ragas_df.head())

                final_ragas_df.to_csv(FINAL_RESULTS_PATH, index=False, encoding='utf-8')
                logging.info(f"Final evaluation results saved to: {FINAL_RESULTS_PATH}")

                faith_cols = [c for c in final_ragas_df.columns if 'faithfulness' in c.lower()]
                if faith_cols:
                    faith_mean = final_ragas_df[faith_cols[0]].mean()
                    logging.info(f"Mean faithfulness: {faith_mean:.3f}")
                    if faith_mean < 0.75:
                        logging.error(f"Faithfulness regression detected: mean {faith_mean:.3f} < 0.75. Failing evaluation.")
                        sys.exit(2)
                else:
                    logging.warning("No faithfulness metric found in final results for regression gate check.")
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
        if os.path.exists(INTERMEDIATE_RESULTS_PATH):
            try:
                intermediate_df = pd.read_csv(
                    INTERMEDIATE_RESULTS_PATH,
                    quotechar='"',
                    engine='python',
                    quoting=csv.QUOTE_NONNUMERIC,
                    doublequote=True
                )
                intermediate_df['question'] = intermediate_df['question'].astype(str)
                processed_questions = set(intermediate_df['question'].tolist())
                logging.info(f"Loaded {len(processed_questions)} already processed questions from {INTERMEDIATE_RESULTS_PATH}")
            except Exception as e:
                logging.warning(f"Could not load or parse intermediate results file {INTERMEDIATE_RESULTS_PATH}. Starting fresh or potentially reprocessing. Error: {e}")
                processed_questions = set()

        questions_processed_this_run = 0
        total_questions_in_gt = len(eval_df)
        new_results_list = []

        for index, row in eval_df.iterrows():
            if index < args.start_index:
                continue

            question = row['question']
            ground_truth = row['ground_truth']

            if str(question) in processed_questions:
                logging.debug(f"Skipping already processed question (Index {index}): {str(question)[:80]}...")
                continue

            if questions_processed_this_run >= args.max_questions:
                logging.info(f"Reached max number of new questions for this run ({args.max_questions}). Stopping.")
                break

            logging.info(f"Processing question {index + 1}/{total_questions_in_gt}: {str(question)[:80]}...")
            answer, contexts = run_system_for_evaluation(question)

            try:
                contexts_str = json.dumps(contexts, ensure_ascii=False)
            except TypeError as e:
                logging.error(f"Could not serialize contexts to JSON for question '{question}': {e}. Saving as empty list string.")
                contexts_str = "[]"

            result_data = {
                "question": question,
                "answer": answer,
                "contexts": contexts_str,
                "ground_truth": ground_truth
            }
            new_results_list.append(result_data)
            processed_questions.add(str(question))
            questions_processed_this_run += 1

        if new_results_list:
            new_results_df = pd.DataFrame(new_results_list)
            try:
                file_exists = os.path.exists(INTERMEDIATE_RESULTS_PATH) and os.path.getsize(INTERMEDIATE_RESULTS_PATH) > 0
                new_results_df.to_csv(
                    INTERMEDIATE_RESULTS_PATH,
                    mode='a',
                    header=not file_exists,
                    index=False,
                    encoding='utf-8',
                    quoting=csv.QUOTE_NONNUMERIC,
                    doublequote=True
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