# app/db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from time import sleep
from app.config import settings

Base = declarative_base()

def _create_engine_with_retry(url: str, attempts: int = 10, delay: float = 2.0):
    last = None
    for _ in range(attempts):
        try:
            return create_engine(url, pool_pre_ping=True, pool_recycle=1800, future=True)
        except Exception as e:
            last = e
            sleep(delay)
    raise last

engine = _create_engine_with_retry(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
