#!/usr/bin/env python3
import os
import json
import numpy as np
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from functools import partial

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
    adjust_weights_for_maturity,
    DEFAULT_BLENDING_WEIGHTS
)
from StormCast.kalman import StormKalmanFilter
from StormCast.forecast import forecast_with_uncertainty
from StormCast.config import MIN_VELOCITY_THRESHOLD, MAX_VELOCITY_THRESHOLD, MOTION_SMOOTHING_WINDOW

LEAD_TIMES = {900: '15 min', 1800: '30 min', 2700: '45 min', 3600: '60 min'}

def evaluate_single_file(file_path, ablation_mode="full"):
    results = [] # list of (lt, mae)
    try:
        with open(file_path, 'r') as f:
            cell_data = json.load(f)
    except: return []
            
    if len(cell_data) < 4: return []
    
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
            kf.predict(dt=dt)
            kf.update(observation=displacements[i], track_history=len(motion_history))
            
            hist_len = len(motion_history)
            if hist_len >= 2:
                h_core = compute_storm_core_height(step['properties'].get('p100EchoTop30'), step['properties'].get('EchoTop50'))
                
                # ABLATION: Adaptive Steering
                if ablation_mode == "no_adaptive_steering":
                    # Fixed 0-6km steering (approximate with 500mb for simplicity)
                    v_mean = env.winds[500]
                else:
                    v_mean = compute_adaptive_steering(env, h_core)
                    
                v_bunkers = compute_bunkers_motion(env, h_core)
                v_obs = smooth_observed_motion(motion_history[-min(len(motion_history), MOTION_SMOOTHING_WINDOW):])
                
                wf_step = step['properties']['wind_field']
                shear = np.sqrt((wf_step['u500'] - wf_step['u850'])**2 + (wf_step['v500'] - wf_step['v850'])**2)
                
                # ABLATION: Maturity Gating
                if ablation_mode == "no_maturity_gating":
                    weights = DEFAULT_BLENDING_WEIGHTS
                else:
                    weights = adjust_weights_for_maturity(h_core, hist_len, shear, mucape=env.mucape)
                
                from StormCast.config import BlendingWeights
                
                # ABLATION: Bunkers Deviation
                if ablation_mode == "no_bunkers":
                    v_bunkers = (0.0, 0.0)
                    total = weights.w_obs + weights.w_mean
                    weights = BlendingWeights(
                        w_obs=weights.w_obs / total,
                        w_mean=weights.w_mean / total,
                        w_bunkers=0.0
                    )
                
                v_final = blend_motion(v_obs, v_mean, v_bunkers, weights)
                
                jitter = calculate_motion_jitter(motion_history[-min(len(motion_history), 10):])
                
                # ABLATION: No Kalman Filter (Initial Position)
                start_x, start_y = kf.x, kf.y
                if ablation_mode == "no_kf":
                    start_x, start_y = displacements[i][0], displacements[i][1]
                
                state = StormState(x=start_x, y=start_y, u=v_final[0], v=v_final[1], 
                                  h_core=h_core, echo_top_30=step['properties'].get('p100EchoTop30', 10.0), track_history=hist_len,
                                  motion_jitter=jitter)
                
                for lt in LEAD_TIMES.keys():
                    elapsed = 0
                    for j in range(i + 1, len(cell_data)):
                        elapsed += cell_data[j].get('dt', 0)
                        if abs(elapsed - lt) <= 150:
                            f = forecast_with_uncertainty(state, lead_times=[elapsed])[0]
                            dist_err = np.sqrt((displacements[j][0] - f.x)**2 + (displacements[j][1] - f.y)**2) / 1000.0
                            results.append((lt, dist_err))
                            break
                        if elapsed > lt + 300: break
    return results

def run_ablation(mode, file_list):
    print(f"Running Ablation Benchmark: {mode} on {len(file_list)} tracks...")
    func = partial(evaluate_single_file, ablation_mode=mode)
    with ProcessPoolExecutor() as executor:
        all_results = executor.map(func, file_list)
    
    stats = {lt: [] for lt in LEAD_TIMES.keys()}
    for file_res in all_results:
        for lt, mae in file_res:
            stats[lt].append(mae)
            
    return {lt: np.mean(val) if val else 0 for lt, val in stats.items()}

def main():
    data_dir = "/home/yuchenwei/StormCast_Data"
    file_list = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.json') and f != 'cell_index.json':
                file_list.append(os.path.join(root, f))
    
    ablations = [
        "full",
        "no_adaptive_steering",
        "no_bunkers",
        "no_maturity_gating",
        "no_kf"
    ]
    
    results = {}
    for mode in ablations:
        results[mode] = run_ablation(mode, file_list)
        
    print("\n" + "="*50)
    print("ABLATION STUDY NUMERICAL RESULTS (MAE in km)")
    print("="*50)
    lt_keys = sorted(LEAD_TIMES.keys())
    header = "Lead Time | " + " | ".join([f"{m:15}" for m in ablations])
    print(header)
    print("-" * len(header))
    for lt in lt_keys:
        row = f"{LEAD_TIMES[lt]:9} | "
        row += " | ".join([f"{results[m][lt]:15.3f}" for m in ablations])
        print(row)
    print("="*50 + "\n")
    
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    lt_keys = sorted(LEAD_TIMES.keys())
    lt_labels = [LEAD_TIMES[k] for k in lt_keys]
    
    # Legend labels
    labels = {
        "full": "Full StormCast (Production)",
        "no_adaptive_steering": "No Height-Adaptive Steering",
        "no_bunkers": "No Bunkers Deviation",
        "no_maturity_gating": "No Maturity Gating",
        "no_kf": "No Kalman Filter (Raw Obs Init)"
    }
    
    colors = {
        "full": "#4f46e5",    # Indigo
        "no_adaptive_steering": "#10b981", # Emerald
        "no_bunkers": "#f59e0b", # Amber
        "no_maturity_gating": "#ec4899", # Pink
        "no_kf": "#94a3b8"     # Slate
    }
    
    for mode in ablations:
        mae_vals = [results[mode][lt] for lt in lt_keys]
        ax.plot(lt_labels, mae_vals, label=labels[mode], color=colors[mode], 
                marker='o', linewidth=2.5, markersize=8)
        
    ax.set_xlabel('Lead Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (km)', fontsize=12, fontweight='bold')
    ax.set_title('StormCast Ablation Study: Impact of Components on MAE', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', frameon=True, fontsize=11)
    
    plt.tight_layout()
    os.makedirs('../plots', exist_ok=True)
    plt.savefig('ablation_study.png', dpi=300, bbox_inches='tight')
    plt.savefig('../plots/ablation_study.png', dpi=300, bbox_inches='tight')
    print("Successfully generated ablation_study.png")

if __name__ == "__main__":
    main()
