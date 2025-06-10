import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import logging

# Globals for lazy initialization
_tokenizer = None
_model = None
_pipe = None

MODEL_PATH = os.getenv(
    "MODEL_SQL2TEXT_AI_DETECTOR_PATH",
    "nerzid/roberta-base-openai-detector-text2sql-approach-2",
)


def load_model():
    """
    Load the model and tokenizer for AI text detection.

    This function uses lazy initialization to load the model and tokenizer
    only when they are first needed. This approach saves memory when the
    model isn't in use.

    Global Variables:
        _tokenizer: The tokenizer for processing input text.
        _model: The pre-trained model for AI text detection.
        _pipe: The pipeline for text classification.

    Returns:
        None
    """
    global _tokenizer, _model, _pipe
    if _tokenizer is None or _model is None or _pipe is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _pipe = pipeline("text-classification", model=_model, tokenizer=_tokenizer)


def is_test_human_generated(test_text: str) -> bool:
    """_summary_
        This one works specifically for everyday-used text such as dialogues, or other usually informal texts
    Args:
        test_text (str): Text to be tested whether it is AI-generated or not

    Returns:
        bool: True if the text is human generated, False otherwise
    """
    model_name = "roberta-base-openai-detector"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)

    # 0 = human, 1 = AI
    ai_prob = probs[0][1].item()
    print(ai_prob)
    return ai_prob < 0.5


def chunk_text(text, max_tokens=512):
    """
    Split the input text into chunks of specified maximum token length.

    Args:
        text (str): The input text to be chunked.
        max_tokens (int, optional): The maximum number of tokens per chunk. Defaults to 512.

    Yields:
        str: Decoded text chunks.
    """
    tokens = _tokenizer.encode(text, add_special_tokens=False)
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i : i + max_tokens]
        yield _tokenizer.decode(chunk, skip_special_tokens=True)


def is_ai_generated(text: str) -> bool:
    """
    Determine if the given text is AI-generated.

    This function uses a pre-trained model to classify whether the input text
    is likely to be AI-generated or human-written. It handles long texts by
    chunking them and aggregating predictions.

    Args:
        text (str): The input text to be classified.

    Returns:
        bool: True if the text is likely AI-generated, False otherwise.
    """
    load_model()  # load the model once if it hasn't loaded already

    predictions = []
    for chunk in chunk_text(text, max_tokens=_tokenizer.model_max_length):
        inputs = _tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=_tokenizer.model_max_length,
        ).to(_pipe.device)

        with torch.no_grad():
            outputs = _pipe.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            label_id = torch.argmax(probs).item()
            label = _pipe.model.config.id2label[label_id].lower()
            score = probs[label_id].item()

        predictions.append((label, score))
        logging.info(f"Chunk: {chunk[:100]}... → {label} ({score:.2f})")

    # Combine results: majority voting weighted by confidence
    ai_score = sum(score for label, score in predictions if "ai" in label)
    human_score = sum(score for label, score in predictions if "human" in label)

    return ai_score >= human_score
