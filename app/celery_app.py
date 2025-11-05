from celery import Celery
from datetime import datetime, timezone

from app.config import settings
from app.models_registery import MODEL_REGISTRY
from app.utils.monitor import get_system_stats
from app.utils.file_ops import save_result
from app.db import SessionLocal
from app.models import EvaluationJob

celery = Celery(
    "model_orchestrator",
    broker=settings.broker_url,
    backend=settings.result_backend,
)

celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

@celery.task(bind=True, name="run_model")
def run_model(self, prompt: str, model_name: str, prompt_id: int):
    db = SessionLocal()
    try:
        job = db.query(EvaluationJob).filter_by(task_id=self.request.id).first()
        if not job:
            job = EvaluationJob(
                task_id=self.request.id,
                prompt_id=prompt_id,
                model_name=model_name,
                state="PENDING"
            )
            db.add(job)
            db.commit()
            db.refresh(job)

        job.state = "STARTED"
        db.commit()

        # Validate model presence in registry
        func = MODEL_REGISTRY.get(model_name)
        if func is None:
            raise ValueError(f"Model '{model_name}' is not registered.")

        stats_before = get_system_stats()

        # Execute model function
        result_payload = func(prompt)

        if not isinstance(result_payload, dict):
            raise ValueError("Model function must return a dict payload.")

        stats_after = get_system_stats()
        merged_result = {**result_payload, "system_stats_before": stats_before, "system_stats_after": stats_after}

        # Persist result artifact and store path
        result_path = save_result(model_name=model_name, prompt=prompt, result=merged_result)

        job.state = "SUCCESS"
        job.result_path = result_path
        job.metrics = merged_result
        job.time_taken = merged_result.get("time_taken")
        job.completed_at = datetime.now(timezone.utc)
        db.commit()

        return {"model": model_name, "prompt": prompt, "task_id": self.request.id}

    except Exception as e:
        try:
            db.rollback()
            job = db.query(EvaluationJob).filter_by(task_id=self.request.id).first()
            if job:
                job.state = "FAILURE"
                job.metrics = {"error": str(e)}
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
        finally:
            raise
    finally:
        db.close()