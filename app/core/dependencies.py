import redis
from app.core.config import settings


def get_redis_client():
    return redis.Redis(
        host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB
    )
