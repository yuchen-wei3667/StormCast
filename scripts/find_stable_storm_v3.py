import os
import json
import numpy as np

def get_jitter(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        if len(data) < 20: return None
        
        motion_history = []
        for step in data:
            dx, dy, dt = step.get('dx', 0), step.get('dy', 0), step.get('dt', 0)
            if dt > 0:
                motion_history.append((dx/dt, dy/dt))
        
        if len(motion_history) < 20: return None
        
        # Calculate jitter for mid-track
        window = motion_history[5:15]
        u_vals = [m[0] for m in window]
        v_vals = [m[1] for m in window]
        jitter = np.sqrt(np.var(u_vals) + np.var(v_vals))
        
        avg_speed = np.mean([np.sqrt(m[0]**2 + m[1]**2) for m in motion_history])
        if avg_speed < 10 or avg_speed > 35: return None
        
        return jitter
    except:
        return None

data_dir = "/home/yuchenwei/StormCast_Data/20260214"
res = []
for f in os.listdir(data_dir):
    if f.endswith('.json'):
        path = os.path.join(data_dir, f)
        j = get_jitter(path)
        if j is not None and j < 3.0: # Very stable
            res.append((j, path))

res.sort()
for j, p in res[:5]:
    print(f"{j:.3f} {p}")
