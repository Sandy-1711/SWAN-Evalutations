# app/models.py
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from sqlalchemy.dialects.postgresql import JSONB  # works on Postgres; fallback to JSON if needed
from app.db import Base

class EvaluationJob(Base):
    __tablename__ = "evaluation_jobs"

    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(Text, nullable=False)
    model_name = Column(String(128), nullable=False, index=True)
    task_id = Column(String(64), unique=True, index=True, nullable=False)
    state = Column(String(32), nullable=False, index=True)  # PENDING/STARTED/SUCCESS/FAILURE
    result_path = Column(Text, nullable=True)
    metrics = Column(JSONB, nullable=True)  # if not using Postgres, use from sqlalchemy import JSON
    time_taken = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True)
