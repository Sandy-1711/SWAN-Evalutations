import psutil
import time

def get_system_stats():
    temp = None
    
    try:
        temps = psutil.sensors_temperatures()
        if "coretemp" in temps:
            temp = temps["coretemp"][0].current
    except Exception:
        pass

    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "temperature": temp,
        "timestamp": time.time(),
    }
