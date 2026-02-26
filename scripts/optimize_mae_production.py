#!/usr/bin/env python3
import os
import json
import numpy as np
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import random

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
from StormCast.config import MIN_VELOCITY_THRESHOLD, MAX_VELOCITY_THRESHOLD, BlendingWeights

def evaluate_params(params, tracks):
    sigma_pos, q_scale, window, alpha, w_obs_base = params
    
    Q_POS_BASE = 10000.0
    Q_VEL_BASE = 144.0
    q_pos = Q_POS_BASE * q_scale
    q_vel = Q_VEL_BASE * q_scale
    
    mae_15 = []
    
    remaining = 1.0 - w_obs_base
    base_w = BlendingWeights(w_obs=w_obs_base, w_mean=remaining/2.0, w_bunkers=remaining/2.0)
    
    for cell_data in tracks:
        if len(cell_data) < 5: continue
        
        props0 = cell_data[0]['properties']
        wf = props0['wind_field']
        env = EnvironmentProfile(
            winds={850: (wf['u850'], wf['v850']), 700: (wf['u700'], wf['v700']), 
                   500: (wf['u500'], wf['v500']), 250: (wf['u250'], wf['v250'])},
            timestamp=datetime.fromisoformat(cell_data[0]['timestamp']),
            mucape=props0.get('MUCAPE')
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
                u, v = dx/dt, dy/dt
                speed = np.sqrt(u**2 + v**2)
                if speed < MIN_VELOCITY_THRESHOLD or speed > MAX_VELOCITY_THRESHOLD: continue
                motion_history.append((u, v))
                
                # Manual Kalman Step mirroring production but with grid-search params
                # Predict
                x_new = kf.state[0] + kf.state[2] * dt
                y_new = kf.state[1] + kf.state[3] * dt
                u_new = kf.state[2]
                v_new = kf.state[3]
                if len(motion_history) > 1:
                    u_prev, v_prev = motion_history[-2]
                    u_new = alpha * kf.state[2] + (1.0 - alpha) * u_prev
                    v_new = alpha * kf.state[3] + (1.0 - alpha) * v_prev
                
                kf.state = np.array([x_new, y_new, u_new, v_new])
                kf.P = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]) @ kf.P @ np.array([[1,0,0,0],[0,1,0,0],[dt,0,1,0],[0,dt,0,1]]) + np.diag([q_pos, q_pos, q_vel, q_vel])
                
                # Update
                H = np.array([[1,0,0,0],[0,1,0,0]])
                scale = 1.0 + 5.0 / max(len(motion_history), 1)
                R = np.diag([(sigma_pos*scale)**2, (sigma_pos*scale)**2])
                z = np.array(displacements[i])
                y_err = z - (H @ kf.state)
                S = H @ kf.P @ H.T + R
                K = kf.P @ H.T @ np.linalg.inv(S)
                kf.state = kf.state + K @ y_err
                kf.P = (np.eye(4) - K @ H) @ kf.P
                
                # Evaluate Operational Skill (from obs 2 onwards)
                if len(motion_history) >= 2:
                    h_core = compute_storm_core_height(step['properties'].get('p100EchoTop30'), step['properties'].get('EchoTop50'))
                    v_obs = smooth_observed_motion(motion_history[-min(len(motion_history), window):])
                    v_mean = compute_adaptive_steering(env, h_core)
                    v_bunkers = compute_bunkers_motion(env, h_core)
                    
                    wf_step = step['properties']['wind_field']
                    shear = np.sqrt((wf_step['u500'] - wf_step['u850'])**2 + (wf_step['v500'] - wf_step['v850'])**2)
                    weights = adjust_weights_for_maturity(h_core, len(motion_history), shear, base_weights=base_w, mucape=env.mucape)
                    v_final = blend_motion(v_obs, v_mean, v_bunkers, weights)
                    
                    elapsed = 0
                    for j in range(i + 1, len(cell_data)):
                        elapsed += cell_data[j].get('dt', 0)
                        if abs(elapsed - 900) <= 150:
                            f_x = kf.state[0] + v_final[0] * elapsed
                            f_y = kf.state[1] + v_final[1] * elapsed
                            err = np.sqrt((displacements[j][0] - f_x)**2 + (displacements[j][1] - f_y)**2) / 1000.0
                            mae_15.append(err)
                            break
                        if elapsed > 1200: break
                        
    return np.mean(mae_15) if mae_15 else 99.0

def run_grid_search_worker(params_batch, tracks):
    results = []
    for params in params_batch:
        mae = evaluate_params(params, tracks)
        results.append((params, mae))
    return results

def main():
    # Load training data
    with open('data_splits/train_files.json', 'r') as f:
        train_files = json.load(f)
    
    random.seed(42)
    sample_files = random.sample(train_files, 600)
    tracks = []
    for f_path in sample_files:
        with open(f_path, 'r') as f:
            tracks.append(json.load(f))
            
    # Search Space
    sigmas = [800, 1000, 1500, 2000]
    q_scales = [0.01, 0.03, 0.05]
    windows = [7, 9, 10]
    alphas = [0.90, 0.95, 0.97]
    w_obs_vals = [0.6, 0.7, 0.8]
    
    combinations = []
    for s in sigmas:
        for q in q_scales:
            for w in windows:
                for a in alphas:
                    for wo in w_obs_vals:
                        combinations.append((s, q, w, a, wo))
                        
    print(f"Starting Operational Grid Search with {len(combinations)} combinations on {len(tracks)} tracks...")
    
    batch_size = 32
    batches = [combinations[i:i + batch_size] for i in range(0, len(combinations), batch_size)]
    
    final_results = []
    with ProcessPoolExecutor() as executor:
        func = partial(run_grid_search_worker, tracks=tracks)
        for batch_res in executor.map(func, batches):
            final_results.extend(batch_res)
            
    final_results.sort(key=lambda x: x[1])
    
    print("\nTop 10 Operational Results (MAE-15):")
    print(f"{'Sigma':<6} | {'Q-Scale':<8} | {'Win':<4} | {'Alpha':<5} | {'W-Obs':<5} | {'MAE-15':<8}")
    print("-" * 55)
    for p, mae in final_results[:10]:
        print(f"{p[0]:<6} | {p[1]:<8} | {p[2]:<4} | {p[3]:<5} | {p[4]:<5.1f} | {mae:<8.3f}")

if __name__ == "__main__":
    from functools import partial
    main()
