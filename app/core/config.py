import os


class Settings:
    SERVICE_HOST: str = os.getenv("SERVICE_HOST", "localhost")
    SERVICE_PORT: int = int(os.getenv("SERVICE_PORT", 8000))
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "lm_studio")
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-4b")
    LLM_API_URL = os.getenv("LLM_API_URL", "http://192.168.56.1:1234/v1")
    DATA_PATH = os.getenv("DATA_PATH", "data")
    CHROMA_DB_DATA_PATH = os.getenv("CHROMA_DB_PATH", DATA_PATH + "/chroma_headers_db")


settings = Settings()
