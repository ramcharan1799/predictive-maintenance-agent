import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_sensor_data(fault_type: str = "bearing") -> pd.DataFrame:
    """Generates realistic sensor data with injected faults."""
    np.random.seed(42)
    n = 200
    timestamps = [datetime.now() - timedelta(minutes=i*5) for i in range(n, 0, -1)]

    if fault_type == "bearing":
        vibration_x = np.random.normal(2.5, 0.3, n)
        vibration_y = np.random.normal(2.3, 0.3, n)
        temperature  = np.random.normal(65, 2, n)
        vibration_x[160:] += np.linspace(0, 6, 40)
        temperature[160:]  += np.linspace(0, 15, 40)
        vibration_x[185:] += np.random.normal(0, 2, 15)

    elif fault_type == "imbalance":
        vibration_x = np.random.normal(1.5, 0.2, n)
        vibration_y = np.random.normal(3.8, 0.4, n)
        temperature  = np.random.normal(60, 1.5, n)
        vibration_y += 0.5 * np.sin(np.linspace(0, 4*np.pi, n))

    elif fault_type == "normal":
        vibration_x = np.random.normal(1.8, 0.2, n)
        vibration_y = np.random.normal(1.9, 0.2, n)
        temperature  = np.random.normal(62, 1.5, n)

    else:
        vibration_x = np.random.normal(2.0, 0.3, n)
        vibration_y = np.random.normal(2.0, 0.3, n)
        temperature  = np.random.normal(63, 2, n)

    rpm     = np.random.normal(1480, 15, n)
    current = np.random.normal(12.5, 0.4, n)
    if fault_type != "normal":
        current[170:] += np.linspace(0, 2, 30)

    df = pd.DataFrame({
        "timestamp":   [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
        "vibration_x": np.round(np.abs(vibration_x), 3),
        "vibration_y": np.round(np.abs(vibration_y), 3),
        "temperature":  np.round(temperature, 2),
        "rpm":          np.round(rpm, 1),
        "current_amp":  np.round(np.abs(current), 3),
    })
    return df
