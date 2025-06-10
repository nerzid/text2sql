import os
from typing import List
import logging
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from app.chromadb.client import get_header_collection
from app.llm.predictors import get_disambiguated_text
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from app.core.constants import (
    IS_TOO_VAGUE_MESSAGE,
    MAX_HEADERS,
    TEXT_TO_SQL_PROMPT_TEMPLATE,
)

MODEL_PATH = os.getenv("MODEL_TEXT2SQL_PATH", "nerzid/qwen2.5-3B-4bit-text2sql")

# Globals for lazy initialization
_tokenizer = None
_model = None
_pipe = None

logger = logging.getLogger(__name__)


collection = get_header_collection()


def load_model():
    """
    Load the model, tokenizer, and pipeline for text-to-SQL generation.

    This function uses lazy initialization to load the components only when they are first needed.
    It sets up global variables for the tokenizer, model, and pipeline.

    Global Variables:
        _tokenizer: The tokenizer for processing input text.
        _model: The pre-trained model for text-to-SQL generation.
        _pipe: The pipeline for text generation.

    Returns:
        None
    """
    global _tokenizer, _model, _pipe
    if _tokenizer is None or _model is None or _pipe is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, trust_remote_code=True, device_map="auto"
        )
        _pipe = pipeline("text-generation", model=_model, tokenizer=_tokenizer)


def get_relevant_headers(question: str, top_k: int = 20) -> List[str]:
    """
    Queries ChromaDB for headers most semantically similar to the input question.

    Args:
        question (str): The user input or question.
        top_k (int): Number of top headers to return.

    Returns:
        List[str]: List of header strings.
    """
    if not collection:
        logger.warning("ChromaDB collection is not available.")
        return []

    try:
        results = collection.query(
            query_texts=[question], n_results=min(top_k, MAX_HEADERS)
        )
        logger.info(f"Found {len(results['documents'][0])} relevant headers.")
        return results["documents"][0] if results["documents"] else []
    except Exception as e:
        logger.error(f"Error querying headers from ChromaDB: {e}")
        return []


def preprocess_text(question: str, top_k_headers: int = 20) -> dict:
    """
    Enhances a vague or ambiguous natural language question by incorporating relevant table headers.

    Args:
        question (str): Natural language question to process.
        top_k_headers (int): Number of relevant headers to include.

    Returns:
        str: Either a more precise prompt for the LLM or a warning message if the question is too vague.
    """
    question = question.strip()
    if not question:
        return "Empty question provided."

    relevant_headers = get_relevant_headers(question, top_k=top_k_headers)
    logger.info(f"Relevant headers: {relevant_headers}")
    try:
        result = get_disambiguated_text(
            text=question, relevant_headers=relevant_headers
        )
        return result
    except Exception as e:
        logger.error(f"Failed to disambiguate question: {e}")
        return question  # Fallback to original question


def text2sql(question: str, top_k_headers: int = 20) -> str:
    """
    Converts a natural language question to a SQL query.

    This function takes a question in natural language, processes it, and generates
    a corresponding SQL query. It uses a pre-trained language model to perform the
    text-to-SQL conversion.

    Args:
        question (str): The natural language question to convert to SQL.
        top_k_headers (int, optional): The number of relevant table headers to consider.
                                       Defaults to 20.

    Returns:
        str: The generated SQL query as a string, or an error message if the question
             is too vague to process.

    Note:
        This function requires a loaded model and access to relevant database headers.
        It also uses a predefined prompt template for generating the SQL query.
    """
    load_model()
    # turned off, because I'm gpu poor :(
    # question = preprocess_text(question, top_k_headers)
    if question == IS_TOO_VAGUE_MESSAGE:
        return question
    headers = get_relevant_headers(question, top_k_headers)
    table_str = _build_table_str(headers)
    prompt = TEXT_TO_SQL_PROMPT_TEMPLATE.format(table_str=table_str, question=question)
    result = _pipe(prompt, max_new_tokens=100)[0]["generated_text"]
    # Remove the prompt prefix
    result = result[len(prompt) :].strip()
    return result


def _build_table_str(headers: list[str]) -> str:
    return " | ".join([f"{h} (text)" for h in headers])
