import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


from app.core.config import settings


def get_chroma_client(
    path: str = settings.CHROMA_DB_DATA_PATH,
) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=path)


def get_header_collection(name: str = "headers"):
    client = get_chroma_client()
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name=name, embedding_function=embedding_function
    )


def upsert_headers(collection, headers: list[str], batch_size: int = 5000):
    for i in range(0, len(headers), batch_size):
        chunk = headers[i : i + batch_size]
        ids = [f"header_{i + j}" for j in range(len(chunk))]
        collection.upsert(documents=chunk, ids=ids)
        print(f"Inserted batch {(i // batch_size) + 1} with {len(chunk)} headers.")

    print(f"Done. Stored {len(headers)} headers in ChromaDB.")
