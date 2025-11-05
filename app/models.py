# app/models.py
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from app.db import Base

class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True, index=True)
    prompt_text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Relationship to evaluation jobs
    evaluation_jobs = relationship("EvaluationJob", back_populates="prompt_record", cascade="all, delete-orphan")

class EvaluationJob(Base):
    __tablename__ = "evaluation_jobs"

    id = Column(Integer, primary_key=True, index=True)
    prompt_id = Column(Integer, ForeignKey("prompts.id"), nullable=False, index=True)
    model_name = Column(String(128), nullable=False, index=True)
    task_id = Column(String(64), unique=True, index=True, nullable=False)
    state = Column(String(32), nullable=False, index=True)  # PENDING/STARTED/SUCCESS/FAILURE
    result_path = Column(Text, nullable=True)
    metrics = Column(JSONB, nullable=True)
    time_taken = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationship back to prompt
    prompt_record = relationship("Prompt", back_populates="evaluation_jobs")