from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from pathlib import Path

from app.schemas import PromptRequest
from app.celery_app import run_model
from app.db import SessionLocal
from app.models import EvaluationJob
from app.config import settings

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/run")
def add_prompt(request: PromptRequest, db: Session = Depends(get_db)):
    prompt = request.prompt
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    # settings.models can be a list or dict of model names. Iterate over keys if dict.
    model_names = settings.models.keys() if isinstance(settings.models, dict) else settings.models
    if not model_names:
        raise HTTPException(status_code=400, detail="No models configured in settings.models.")

    task_ids = {}
    for model in model_names:
        # enqueue task
        task = run_model.delay(prompt, model)

        # persist job row
        job = EvaluationJob(
            prompt=prompt,
            model_name=model,
            task_id=task.id,
            state="PENDING",
        )
        try:
            db.add(job)
            db.commit()
            db.refresh(job)
        except Exception:
            db.rollback()
            raise

        task_ids[model] = task.id

    return {"message": "Prompt queued", "prompt": prompt, "task_ids": task_ids}


@router.get("/status/{task_id}")
def get_status(task_id: str, db: Session = Depends(get_db)):
    job = db.query(EvaluationJob).filter_by(task_id=task_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": job.task_id,
        "model": job.model_name,
        "state": job.state,
        "metrics": job.metrics,
        "completed_at": job.completed_at,
    }


@router.get("/results")
def list_results(db: Session = Depends(get_db)):
    results = (
        db.query(EvaluationJob)
        .filter(EvaluationJob.state == "SUCCESS")
        .order_by(EvaluationJob.completed_at.desc())
        .all()
    )
    return [
        {
            "task_id": r.task_id,
            "model": r.model_name,
            "prompt": r.prompt,
            "result_path": r.result_path,
            "completed_at": r.completed_at,
        }
        for r in results
    ]


@router.get("/download/{task_id}")
def download_result(task_id: str, db: Session = Depends(get_db)):
    job = db.query(EvaluationJob).filter_by(task_id=task_id).first()
    if not job or not job.result_path:
        raise HTTPException(status_code=404, detail="Result not found")

    path = Path(job.result_path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Result file missing on disk")

    # If your saved result is JSON text and you prefer JSONResponse:
    if path.suffix.lower() == ".json":
        try:
            return JSONResponse(content=path.read_text(encoding="utf-8"))
        except Exception:
            # Fallback to raw file
            return FileResponse(str(path), media_type="application/json", filename=path.name)

    # Otherwise return as a file download
    return FileResponse(str(path), filename=path.name)
