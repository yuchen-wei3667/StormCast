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
        
        if len(motion_history) < 15: return None
        
        # Calculate jitter like StormCast.blending.calculate_motion_jitter
        # Using a 10-point window at the "sweet spot"
        window = motion_history[5:15]
        u_vals = [m[0] for m in window]
        v_vals = [m[1] for m in window]
        u_var = np.var(u_vals)
        v_var = np.var(v_vals)
        jitter = np.sqrt(u_var + v_var)
        
        # Also check speed to avoid stationary cells
        avg_speed = np.mean([np.sqrt(m[0]**2 + m[1]**2) for m in window])
        if avg_speed < 8: return None # Keep it interesting
        
        return jitter
    except:
        return None

data_dir = "/home/yuchenwei/StormCast_Data"
results = []

for root, _, files in os.walk(data_dir):
    for f in files:
        if f.endswith('.json') and f != 'cell_index.json':
            full_path = os.path.join(root, f)
            jitter = get_jitter(full_path)
            if jitter is not None:
                results.append((jitter, full_path))
    if len(results) > 1000: break

results.sort()
for j, f in results[:5]:
    print(f"{j:.3f} {f}")
