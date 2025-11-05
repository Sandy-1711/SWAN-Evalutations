from app.services import baseline_service, chained_service, rag_chained_service

MODEL_REGISTRY = {
    "baseline": lambda prompt: baseline_service.run_pipeline(prompt),
    "chained": lambda prompt: chained_service.run_pipeline(prompt),
    "rag_chained_k3t5": lambda prompt: rag_chained_service.run_pipeline(prompt, top_k=3, distance_threshold=0.5),
    "rag_chained_k1t2": lambda prompt: rag_chained_service.run_pipeline(prompt, top_k=1, distance_threshold=0.2),
    "rag_chained_k1t5": lambda prompt: rag_chained_service.run_pipeline(prompt, top_k=1, distance_threshold=0.5),
}
