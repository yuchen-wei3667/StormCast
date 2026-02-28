import os
import json
import numpy as np

data_dir = "/home/yuchenwei/StormCast_Data/20260214"
files = os.listdir(data_dir)
results = []

for f in files[:2000]: # Check a subset
    if not f.endswith('.json'): continue
    path = os.path.join(data_dir, f)
    try:
        with open(path, 'r') as jf:
            data = json.load(jf)
        if len(data) < 15: continue
        
        m_hist = []
        for s in data:
            dt = s.get('dt', 0)
            if dt > 0: m_hist.append((s.get('dx',0)/dt, s.get('dy',0)/dt))
        
        if len(m_hist) < 10: continue
        
        # Jitter of last 10 points
        window = m_hist[-10:]
        u_vals = [m[0] for m in window]
        v_vals = [m[1] for m in window]
        jitter = np.sqrt(np.var(u_vals) + np.var(v_vals))
        
        results.append((jitter, path))
    except:
        continue

results.sort()
for j, p in results[:10]:
    print(f"{j:.3f} {p}")
