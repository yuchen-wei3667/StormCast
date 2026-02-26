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
    blend_motion,
    adjust_weights_for_maturity
)
from StormCast.config import MIN_VELOCITY_THRESHOLD, MAX_VELOCITY_THRESHOLD, BlendingWeights

def validate_config(sigma_pos, q_scale, window, alpha, w_obs_base, label):
    with open('data_splits/test_files.json', 'r') as f:
        test_files = json.load(f)
        
    print(f"Validating {label} on {len(test_files)} test tracks...")
    
    Q_POS_BASE = 10000.0
    Q_VEL_BASE = 144.0
    q_pos = Q_POS_BASE * q_scale
    q_vel = Q_VEL_BASE * q_scale
    
    mae_15 = []
    
    remaining = 1.0 - w_obs_base
    base_w = BlendingWeights(w_obs=w_obs_base, w_mean=remaining/2.0, w_bunkers=remaining/2.0)
    
    for file_path in test_files:
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
        P = np.array([[q_pos, 0, 0, 0], [0, q_pos, 0, 0], [0, 0, q_vel*3, 0], [0, 0, 0, q_vel*3]])
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
                
                # PREDICT
                x_new = state[0] + state[2] * dt
                y_new = state[1] + state[3] * dt
                u_new = state[2]
                v_new = state[3]
                if prev_velocity is not None:
                    u_new = alpha * u_new + (1.0 - alpha) * prev_velocity[0]
                    v_new = alpha * v_new + (1.0 - alpha) * prev_velocity[1]
                state = np.array([x_new, y_new, u_new, v_new])
                prev_velocity = (u_new, v_new)
                P = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]) @ P @ np.array([[1,0,0,0],[0,1,0,0],[dt,0,1,0],[0,dt,0,1]]) + np.diag([q_pos, q_pos, q_vel, q_vel])
                
                # UPDATE
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
                        
    return np.mean(mae_15)

if __name__ == "__main__":
    mae_a = validate_config(1000.0, 0.01, 9, 0.95, 0.7, "Config A (Sigma=1000)")
    mae_b = validate_config(800.0, 0.01, 9, 0.95, 0.7, "Config B (Sigma=800)")
    
    print("\nFinal Test Set MAE-15:")
    print(f"Config A (Sigma=1000): {mae_a:.4f} km")
    print(f"Config B (Sigma=800): {mae_b:.4f} km")
