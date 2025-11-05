from app.services import baseline_service

MODEL_REGISTRY = {
    "baseline": baseline_service.generate_response,
}
