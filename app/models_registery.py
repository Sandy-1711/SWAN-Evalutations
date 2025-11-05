from app.services import baseline_service
from app.services import chained_service
from app.services import rag_chained_service
MODEL_REGISTRY = {
    "baseline": baseline_service.run_pipeline,
    "chained": chained_service.run_pipeline,
    "rag_chained": rag_chained_service.run_pipeline,
}
