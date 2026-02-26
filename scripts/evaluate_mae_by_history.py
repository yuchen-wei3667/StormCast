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
    adjust_weights_for_maturity
)
from StormCast.kalman import StormKalmanFilter
from StormCast.forecast import forecast_with_uncertainty
from StormCast.config import MIN_VELOCITY_THRESHOLD, MAX_VELOCITY_THRESHOLD, MOTION_SMOOTHING_WINDOW

CHI2_95 = 5.991
LEAD_TIMES = {900: '15 min', 1800: '30 min', 2700: '45 min', 3600: '60 min'}
MAX_HISTORY_LEN = 50

def evaluate_single_file(file_path):
    results = [] # list of (lt, hist_len, hit, mae, sigma)
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
        freezing_level_km=props0.get('freezing_level_height'),
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
            if 2 <= hist_len <= MAX_HISTORY_LEN:
                h_core = compute_storm_core_height(step['properties'].get('p100EchoTop30'), step['properties'].get('EchoTop50'))
                v_mean = compute_adaptive_steering(env, h_core)
                v_bunkers = compute_bunkers_motion(env, h_core)
                
                v_obs = smooth_observed_motion(motion_history[-min(len(motion_history), MOTION_SMOOTHING_WINDOW):])
                jitter = calculate_motion_jitter(motion_history[-min(len(motion_history), 10):])
                
                wf = step['properties']['wind_field']
                shear = np.sqrt((wf['u500'] - wf['u850'])**2 + (wf['v500'] - wf['v850'])**2)
                weights = adjust_weights_for_maturity(h_core, hist_len, shear, mucape=env.mucape)
                v_final = blend_motion(v_obs, v_mean, v_bunkers, weights)
                
                state = StormState(x=kf.x, y=kf.y, u=v_final[0], v=v_final[1], 
                                  h_core=h_core, echo_top_30=step['properties'].get('p100EchoTop30', 10.0), track_history=hist_len,
                                  motion_jitter=jitter)
                
                for target_lt in LEAD_TIMES.keys():
                    elapsed = 0
                    for j in range(i + 1, len(cell_data)):
                        elapsed += cell_data[j].get('dt', 0)
                        if abs(elapsed - target_lt) <= 150:
                            f = forecast_with_uncertainty(state, lead_times=[elapsed])[0]
                            dist_err = np.sqrt((displacements[j][0] - f.x)**2 + (displacements[j][1] - f.y)**2) / 1000.0
                            hit = 0
                            if f.sigma_x > 0 and f.sigma_y > 0:
                                val = ((displacements[j][0] - f.x)**2 / f.sigma_x**2) + ((displacements[j][1] - f.y)**2 / f.sigma_y**2)
                                if val <= CHI2_95: hit = 1
                            results.append((target_lt, hist_len, hit, dist_err, (f.sigma_x + f.sigma_y)/2000.0 * np.sqrt(CHI2_95)))
                            break
                        if elapsed > target_lt + 300: break
    return results

def main():
    data_dir = "/home/yuchenwei/StormCast_Data"
    file_list = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.json') and f != 'cell_index.json':
                file_list.append(os.path.join(root, f))
                
    print(f"Evaluating EVERYTHING for MAE vs history: {len(file_list)} tracks...")
    
    # lt -> hist_len -> {'hits': 0, 'total': 0, 'mae': [], 'sigmas': []}
    stats = {lt: {h: {'hits': 0, 'total': 0, 'mae': [], 'sigmas': []} for h in range(2, MAX_HISTORY_LEN + 1)} for lt in LEAD_TIMES.keys()}
    
    with ProcessPoolExecutor() as executor:
        all_results = executor.map(evaluate_single_file, file_list)
        
    for file_results in all_results:
        for lt, h, hit, mae, sigma in file_results:
            stats[lt][h]['hits'] += hit
            stats[lt][h]['total'] += 1
            stats[lt][h]['mae'].append(mae)
            stats[lt][h]['sigmas'].append(sigma)
            
    print("Massive Evaluation complete.")
    
    # Plotting
    plt.style.use('seaborn-v0_8-white')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    colors = {900: '#2563eb', 1800: '#d97706', 2700: '#16a34a', 3600: '#dc2626'}
    
    valid_histories = []
    sample_sizes = []
    for h in range(2, MAX_HISTORY_LEN + 1):
        if stats[900][h]['total'] >= 100:
            valid_histories.append(h)
            sample_sizes.append(sum(stats[lt][h]['total'] for lt in LEAD_TIMES) / 4.0)
            
    for lt, label in LEAD_TIMES.items():
        maes = [np.mean(stats[lt][h]['mae']) if stats[lt][h]['total'] > 0 else np.nan for h in valid_histories]
        ax1.plot(valid_histories, maes, label=label, color=colors[lt], linewidth=2.0, marker='.', markersize=8)

    ax2 = ax1.twinx()
    ax2.fill_between(valid_histories, sample_sizes, color='#9ca3af', alpha=0.3, label='Sample Size')
    ax2.set_ylabel('Observations used (Sample Depth)', fontsize=10, color='#4b5563', fontweight='bold')
    ax2.tick_params(axis='y', colors='#4b5563', labelsize=9)
    ax2.grid(False)

    ax1.set_xlabel('Filter History Length (Observations)', fontsize=11, color='#374151', fontweight='bold')
    ax1.set_ylabel('Mean Absolute Error (km)', fontsize=11, color='#374151', fontweight='bold')
    from matplotlib.ticker import MaxNLocator
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_ylim(bottom=0)
    ax1.spines['top'].set_visible(False)
    ax1.legend(loc='upper right', frameon=True, fontsize=10, facecolor='white', framealpha=0.9)
    plt.title('StormCast MAE vs. Tracking History (Total Population)', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    os.makedirs('../plots', exist_ok=True)
    plt.savefig('mae_vs_history.png', dpi=300, bbox_inches='tight')
    plt.savefig('../plots/mae_vs_history.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
