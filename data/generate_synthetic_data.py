
# data/generate_synthetic_data.py
import os, argparse, random
import numpy as np, pandas as pd
from datetime import datetime, timedelta

def generate(num_machines=5, hours=72, freq_min=1, anomaly_prob=0.005, seed=42, outpath="data/synthetic_sensor_data.csv"):
    np.random.seed(seed)
    random.seed(seed)
    rows = []
    start = datetime.utcnow()
    steps = int(hours * 60 / freq_min)
    machine_types = ["loom","milling","welding","lathe","compressor"]
    for m in range(num_machines):
        machine_id = f"machine_{m+1}"
        mtype = machine_types[m % len(machine_types)]
        baseline_current = np.clip(np.random.normal(15, 3), 5, 40)
        baseline_temp = np.clip(np.random.normal(45, 5), 20, 90)
        baseline_vib = int(np.clip(np.random.normal(50, 8), 10, 200))
        for t in range(steps):
            ts = start + timedelta(minutes=freq_min*t)
            hour = ts.hour + ts.minute/60.0
            solar_peak = max(0, 5 * (1 - ((hour-12)/6)**2))
            drift = 1 + 0.0005 * t * (1 + (m%3))
            current = max(0.5, np.random.normal(baseline_current * drift, 0.8))
            temp = np.random.normal(baseline_temp * (1 + 0.0003*t), 0.6)
            vibration = int(np.random.normal(baseline_vib, 6))
            power_kw = round(current * 0.24, 3)
            solar_kw = round(solar_peak + np.random.normal(0, 0.2), 3)
            label = 0
            if np.random.rand() < anomaly_prob or (t%500==0 and t>0):
                if np.random.rand() < 0.6:
                    vibration += int(np.random.uniform(60, 200))
                    current += np.random.uniform(3, 8)
                    temp += np.random.uniform(2, 6)
                    label = 1
                else:
                    drift_mult = 1 + np.random.uniform(0.05, 0.2)
                    current *= drift_mult
                    temp *= drift_mult
                    vibration += int(np.random.uniform(20,60))
                    label = 1
            rows.append({
                "timestamp": ts.isoformat(),
                "machine_id": machine_id,
                "machine_type": mtype,
                "current_a": round(current, 3),
                "temperature_c": round(temp, 2),
                "vibration_raw": int(vibration),
                "power_kw": power_kw,
                "solar_kw": round(max(0, solar_kw),3),
                "label": label
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_csv(outpath, index=False)
    print(f"Wrote {len(df)} rows to {outpath}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--machines", type=int, default=5)
    parser.add_argument("--hours", type=int, default=72)
    parser.add_argument("--freq_min", type=int, default=1)
    parser.add_argument("--out", type=str, default="data/synthetic_sensor_data.csv")
    args = parser.parse_args()
    generate(num_machines=args.machines, hours=args.hours, freq_min=args.freq_min, outpath=args.out)
