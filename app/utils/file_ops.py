import json
import os
from datetime import datetime
from app.config import settings

RESULTS_DIR = settings.results_dir

def save_result(model_name, prompt, result):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(RESULTS_DIR, f"{model_name}_{timestamp}.json")

    # extract system stats safely
    stats_after = result.get("system_stats_after", {})
    
    result_json = {
        "model": model_name,
        "prompt": prompt,
        "code": result.get("code"),
        "output": result.get("output"),
        "time_taken": result.get("time_taken"),
        "cpu_percent": stats_after.get("cpu_percent"),
        "memory_percent": stats_after.get("memory_percent"),
        "temperature": stats_after.get("temperature"),
    }

    with open(file_path, "w") as f:
        json.dump(result_json, f, indent=4)

    return file_path
