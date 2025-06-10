from app.chromadb.client import get_chroma_client, get_header_collection, upsert_headers


def load_headers_from_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    headers = load_headers_from_file("unique_headers.txt")
    client = get_chroma_client()
    collection = get_header_collection(client)
    upsert_headers(collection, headers)


if __name__ == "__main__":
    main()
