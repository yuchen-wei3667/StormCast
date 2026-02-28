import os
import json
import numpy as np

def analyze_track(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        if len(data) < 20: return None
        
        velocities = []
        for step in data:
            dt = step.get('dt', 0)
            if dt > 0:
                velocities.append((step.get('dx', 0)/dt, step.get('dy', 0)/dt))
        
        if len(velocities) < 15: return None
        
        vel_arr = np.array(velocities)
        speed = np.linalg.norm(vel_arr, axis=1)
        
        # Filter for reasonable speeds
        if np.mean(speed) < 5 or np.mean(speed) > 40: return None
        
        # Jitter as std of speed and direction
        speed_std = np.std(speed)
        
        # Direction jitter
        angles = np.arctan2(vel_arr[:, 1], vel_arr[:, 0])
        angle_std = np.std(np.unwrap(angles))
        
        # Total metric: low speed std and low angle std
        score = speed_std + abs(angle_std) * 10 
        return score
    except:
        return None

data_dir = "/home/yuchenwei/StormCast_Data"
best_file = None
best_score = float('inf')

count = 0
for root, _, files in os.walk(data_dir):
    for f in files:
        if f.endswith('.json') and f != 'cell_index.json':
            full_path = os.path.join(root, f)
            score = analyze_track(full_path)
            if score is not None and score < best_score:
                best_score = score
                best_file = full_path
            count += 1
            if count > 5000: break # Sample enough
    if count > 5000: break

print(best_file)
