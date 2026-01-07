#!/usr/bin/env python3
import os
import json
import numpy as np
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from StormCast.types import StormState, EnvironmentProfile
from StormCast.diagnostics import (
    compute_storm_core_height,
    compute_adaptive_steering,
    compute_bunkers_motion
)
from StormCast.blending import (
    smooth_observed_motion,
    calculate_motion_jitter,
    blend_motion,
    adjust_weights_for_maturity
)
from StormCast.kalman import StormKalmanFilter
from StormCast.forecast import forecast_with_uncertainty

def evaluate_coverage(data_dir, max_files=1000):
    lead_times = [900, 1800, 2700, 3600] # 15, 30, 45, 60 min
    stats = {lt: {'hits': 0, 'total': 0} for lt in lead_times}
    
    file_list = []
    for date_folder in sorted(os.listdir(data_dir)):
        date_path = os.path.join(data_dir, date_folder)
        if not os.path.isdir(date_path): continue
        for json_file in os.listdir(date_path):
            if json_file.endswith('.json') and json_file != 'cell_index.json':
                file_list.append(os.path.join(date_path, json_file))
    
    import random
    random.seed(42)
    random.shuffle(file_list)
    file_list = file_list[:max_files]
    
    CHI2_95 = 5.991
    
    print(f"Evaluating {len(file_list)} files with jitter support...")
    
    for file_path in file_list:
        with open(file_path, 'r') as f:
            cell_data = json.load(f)
        
        if len(cell_data) < 10: continue
        
        # Prepare environment
        props0 = cell_data[0]['properties']
        env = EnvironmentProfile(
            winds={850: (props0['u850'], props0['v850']), 700: (props0['u700'], props0['v700']), 
                   500: (props0['u500'], props0['v500']), 250: (props0['u250'], props0['v250'])},
            timestamp=datetime.fromisoformat(cell_data[0]['timestamp'])
        )
        
        kf = StormKalmanFilter(initial_state=[0.0, 0.0, 0.0, 0.0])
        motion_history = []
        cur_x, cur_y = 0.0, 0.0
        displacements = []
        for step in cell_data:
            cur_x += step.get('dx', 0)
            cur_y += step.get('dy', 0)
            displacements.append((cur_x, cur_y))
            
        for i, step in enumerate(cell_data):
            dx, dy, dt = step.get('dx', 0), step.get('dy', 0), step.get('dt', 0)
            if dt > 0:
                motion_history.append((dx/dt, dy/dt))
                kf.predict(dt=dt)
                kf.update(observation=displacements[i], track_history=len(motion_history))
                
                if len(motion_history) >= 5:
                    h_core = compute_storm_core_height(step['properties']['EchoTop30'], step['properties']['EchoTop50'])
                    v_mean = compute_adaptive_steering(env, h_core)
                    v_bunkers = compute_bunkers_motion(env, h_core)
                    v_obs = smooth_observed_motion(motion_history[-5:])
                    jitter = calculate_motion_jitter(motion_history[-10:])
                    
                    props = step['properties']
                    shear = np.sqrt((props['u500'] - props['u850'])**2 + (props['v500'] - props['v850'])**2)
                    weights = adjust_weights_for_maturity(h_core, len(motion_history), shear)
                    v_final = blend_motion(v_obs, v_mean, v_bunkers, weights)
                    
                    state = StormState(x=kf.x, y=kf.y, u=v_final[0], v=v_final[1], 
                                      h_core=h_core, track_history=len(motion_history),
                                      motion_jitter=jitter)
                    
                    for target_lt in lead_times:
                        elapsed = 0
                        for j in range(i + 1, len(cell_data)):
                            elapsed += cell_data[j].get('dt', 0)
                            if abs(elapsed - target_lt) <= 150:
                                f = forecast_with_uncertainty(state, lead_times=[elapsed])[0]
                                act_x, act_y = displacements[j]
                                if f.sigma_x > 0 and f.sigma_y > 0:
                                    val = ((act_x - f.x)**2 / f.sigma_x**2) + ((act_y - f.y)**2 / f.sigma_y**2)
                                    if val <= CHI2_95: stats[target_lt]['hits'] += 1
                                    stats[target_lt]['total'] += 1
                                break
                            if elapsed > target_lt + 300: break
    return stats

if __name__ == "__main__":
    results = evaluate_coverage("/home/yuchenwei/StormCast/StormCast_Training_Data", max_files=200)
    print("\nAccuracy with Dynamic Jitter Uncertainty:")
    for lt in sorted(results.keys()):
        h, t = results[lt]['hits'], results[lt]['total']
        print(f"{lt//60:>2} min Coverage: {(h/t*100):>6.1f}% ({h}/{t})")
