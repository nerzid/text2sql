import json
from typing import Set


def extract_unique_headers_from_file(jsonl_path: str) -> Set[str]:
    """
    Extracts a set of unique headers from a JSONL file of tables.

    Args:
        jsonl_path (str): Path to the JSONL file.

    Returns:
        Set[str]: A set of unique column headers.
    """
    unique_headers = set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            table = json.loads(line)
            headers = table.get("header", [])
            unique_headers.update(headers)

    return unique_headers


def save_headers(headers: Set[str], output_path: str) -> None:
    """
    Saves the sorted list of headers to a file.

    Args:
        headers (Set[str]): The set of unique headers.
        output_path (str): Where to write them.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for header in sorted(headers):
            f.write(header + "\n")
