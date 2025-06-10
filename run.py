import logging

# Configure logging to become visible in uvicorn
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

import litellm

litellm._logging._disable_debugging()

import uvicorn
from app.main import app
from app.core.config import settings


if __name__ == "__main__":
    uvicorn.run(app, host=settings.SERVICE_HOST, port=settings.SERVICE_PORT)
