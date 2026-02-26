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
    blend_motion,
    adjust_weights_for_maturity
)
from StormCast.config import MIN_VELOCITY_THRESHOLD, MAX_VELOCITY_THRESHOLD, BlendingWeights

def evaluate_params(args):
    files, sigma_pos, q_scale, window, alpha, w_obs_base = args
    Q_POS_BASE = 10000.0
    Q_VEL_BASE = 144.0
    mae_15 = []
    remaining = 1.0 - w_obs_base
    base_w = BlendingWeights(w_obs=w_obs_base, w_mean=remaining/2.0, w_bunkers=remaining/2.0)
    for file_path in files:
        with open(file_path, 'r') as f:
            cell_data = json.load(f)
        if len(cell_data) < 5: continue
        props0 = cell_data[0]['properties']
        wf = props0['wind_field']
        env = EnvironmentProfile(
            winds={850: (wf['u850'], wf['v850']), 700: (wf['u700'], wf['v700']), 
                   500: (wf['u500'], wf['v500']), 250: (wf['u250'], wf['v250'])},
            timestamp=datetime.fromisoformat(cell_data[0]['timestamp'])
        )
        state = np.array([0.0, 0.0, 0.0, 0.0])
        P = np.array([[Q_POS_BASE*q_scale,0,0,0],[0,Q_POS_BASE*q_scale,0,0],[0,0,Q_VEL_BASE*q_scale*3,0],[0,0,0,Q_VEL_BASE*q_scale*3]])
        prev_velocity = None
        displacements = []
        cur_x, cur_y = 0.0, 0.0
        for step in cell_data:
            cur_x += step.get('dx', 0)
            cur_y += step.get('dy', 0)
            displacements.append((cur_x, cur_y))
        motion_history = []
        for i, step in enumerate(cell_data):
            dx, dy, dt = step.get('dx', 0), step.get('dy', 0), step.get('dt', 0)
            if dt > 0:
                u, v = dx/dt, dy/dt
                speed = np.sqrt(u**2 + v**2)
                if speed < MIN_VELOCITY_THRESHOLD or speed > MAX_VELOCITY_THRESHOLD: continue
                motion_history.append((u, v))
                x_new = state[0] + state[2] * dt
                y_new = state[1] + state[3] * dt
                u_new = state[2]
                v_new = state[3]
                if prev_velocity is not None:
                    u_new = alpha * u_new + (1.0 - alpha) * prev_velocity[0]
                    v_new = alpha * v_new + (1.0 - alpha) * prev_velocity[1]
                state = np.array([x_new, y_new, u_new, v_new])
                prev_velocity = (u_new, v_new)
                F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
                Q = np.diag([Q_POS_BASE*q_scale, Q_POS_BASE*q_scale, Q_VEL_BASE*q_scale, Q_VEL_BASE*q_scale])
                P = F @ P @ F.T + Q
                H = np.array([[1,0,0,0],[0,1,0,0]])
                scale = 1.0 + 5.0 / max(len(motion_history), 1)
                sigma = sigma_pos * scale
                R = np.diag([sigma**2, sigma**2])
                z = np.array(displacements[i])
                y_err = z - (H @ state)
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                state = state + K @ y_err
                P = (np.eye(4) - K @ H) @ P
                if len(motion_history) >= window:
                    h_core = compute_storm_core_height(step['properties'].get('p100EchoTop30'), step['properties'].get('EchoTop50'))
                    v_obs = smooth_observed_motion(motion_history[-window:])
                    v_mean = compute_adaptive_steering(env, h_core)
                    v_bunkers = compute_bunkers_motion(env, h_core)
                    weights = adjust_weights_for_maturity(h_core, len(motion_history), 15.0, base_weights=base_w)
                    v_final = blend_motion(v_obs, v_mean, v_bunkers, weights)
                    elapsed = 0
                    for j in range(i + 1, len(cell_data)):
                        elapsed += cell_data[j].get('dt', 0)
                        if abs(elapsed - 900) <= 150:
                            f_x = state[0] + v_final[0] * elapsed
                            f_y = state[1] + v_final[1] * elapsed
                            err = np.sqrt((displacements[j][0] - f_x)**2 + (displacements[j][1] - f_y)**2) / 1000.0
                            mae_15.append(err)
                            break
                        if elapsed > 1200: break
    return {'sigma_pos': sigma_pos, 'q_scale': q_scale, 'window': window, 'alpha': alpha, 'w_obs': w_obs_base, 'mae_15': np.mean(mae_15) if mae_15 else 999.0}

def main():
    with open('data_splits/train_files.json', 'r') as f:
        train_files = json.load(f)
    subset_size = 500
    opt_files = random.sample(train_files, subset_size)
    sigmas = [500, 1000, 1500]
    q_scales = [0.01, 0.03, 0.05]
    windows = [5, 7, 9, 11]
    alphas = [0.95, 1.0]
    w_obs_bases = [0.7, 0.8, 0.9]
    tasks = []
    for s in sigmas:
        for q in q_scales:
            for w in windows:
                for a in alphas:
                    for wo in w_obs_bases:
                        tasks.append((opt_files, s, q, w, a, wo))
    print(f"Starting Grid Search Round 4 (MAE-15 Priority) with {len(tasks)} combinations...")
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(evaluate_params, tasks))
    results.sort(key=lambda x: x['mae_15'])
    print("\nTop 10 Results (Sorted by MAE-15):")
    print(f"{'Sigma':<6} | {'Q-Scale':<8} | {'Win':<4} | {'Alpha':<5} | {'W-Obs':<5} | {'MAE-15':<8}")
    print("-" * 60)
    for r in results[:10]:
        print(f"{r['sigma_pos']:<6} | {r['q_scale']:<8} | {r['window']:<4} | {r['alpha']:<5.2f} | {r['w_obs']:<5.1f} | {r['mae_15']:<8.3f}")
    best = results[0]
    print(f"\nWinner: Sigma={best['sigma_pos']}, Q-Scale={best['q_scale']}, Window={best['window']}, Alpha={best['alpha']}, W-Obs={best['w_obs']}")

if __name__ == "__main__":
    main()
