import time
import random


def generate_response(prompt: str):
    start = time.time()
    time.sleep(1)
    end = time.time()
    return {
        "code": f"# Simulated code for prompt: {prompt}",
        "output": {"result": "ok", "details": f"Processed by model_a"},
        "time_taken": end - start,
    }
