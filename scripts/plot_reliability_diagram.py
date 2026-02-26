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

LEAD_TIMES = {900: '15 min', 1800: '30 min', 2700: '45 min', 3600: '60 min'}

def evaluate_single_file(file_path):
    # Returns list of (lt, mahalanobis_dist_sq)
    dists = []
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
                v_mean = compute_adaptive_steering(env, h_core)
                v_bunkers = compute_bunkers_motion(env, h_core)
                v_obs = smooth_observed_motion(motion_history[-min(len(motion_history), MOTION_SMOOTHING_WINDOW):])
                
                wf_step = step['properties']['wind_field']
                shear = np.sqrt((wf_step['u500'] - wf_step['u850'])**2 + (wf_step['v500'] - wf_step['v850'])**2)
                weights = adjust_weights_for_maturity(h_core, hist_len, shear, mucape=env.mucape)
                v_final = blend_motion(v_obs, v_mean, v_bunkers, weights)
                
                jitter = calculate_motion_jitter(motion_history[-min(len(motion_history), 10):])
                state = StormState(x=kf.x, y=kf.y, u=v_final[0], v=v_final[1], 
                                  h_core=h_core, echo_top_30=step['properties'].get('p100EchoTop30', 10.0), track_history=hist_len,
                                  motion_jitter=jitter)
                
                for lt in LEAD_TIMES.keys():
                    elapsed = 0
                    for j in range(i + 1, len(cell_data)):
                        elapsed += cell_data[j].get('dt', 0)
                        if abs(elapsed - lt) <= 150:
                            f = forecast_with_uncertainty(state, lead_times=[elapsed])[0]
                            if f.sigma_x > 0 and f.sigma_y > 0:
                                d2 = ((displacements[j][0] - f.x)**2 / f.sigma_x**2) + ((displacements[j][1] - f.y)**2 / f.sigma_y**2)
                                dists.append((lt, d2))
                            break
                        if elapsed > lt + 300: break
    return dists

def main():
    data_dir = "/home/yuchenwei/StormCast_Data"
    file_list = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.json') and f != 'cell_index.json':
                file_list.append(os.path.join(root, f))
    
    print(f"Generating Reliability Data: {len(file_list)} tracks...")
    
    with ProcessPoolExecutor() as executor:
        all_results = executor.map(evaluate_single_file, file_list)
    
    all_d2 = {lt: [] for lt in LEAD_TIMES.keys()}
    for file_res in all_results:
        for lt, d2 in file_res:
            all_d2[lt].append(d2)
            
    print("Evaluation complete. Calculating Hit Rates...")
    
    # Nominal confidence levels 1% to 99%
    nominal = np.linspace(0.01, 0.99, 50)
    # Thresholds for df=2 Chi-squared: x = -2 * ln(1 - P)
    thresholds = -2.0 * np.log(1.0 - nominal)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1:1 Diagonal
    ax.plot([0, 1], [0, 1], color='#64748b', linestyle='--', linewidth=1.5, label='Perfect Calibration')
    
    colors = {900: '#2563eb', 1800: '#d97706', 2700: '#16a34a', 3600: '#dc2626'}
    
    for lt in sorted(LEAD_TIMES.keys()):
        d2_vals = np.array(all_d2[lt])
        if len(d2_vals) == 0: continue
        
        observed_freq = []
        for t in thresholds:
            hit_rate = np.mean(d2_vals <= t)
            observed_freq.append(hit_rate)
            
        ax.plot(nominal, observed_freq, label=f'StormCast ({LEAD_TIMES[lt]})', color=colors[lt], linewidth=2.5)

    ax.set_xlabel('Nominal Confidence Level', fontsize=12, fontweight='bold', color='#1e293b')
    ax.set_ylabel('Observed Inclusion Frequency', fontsize=12, fontweight='bold', color='#1e293b')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    from matplotlib.ticker import PercentFormatter
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    ax.legend(loc='upper left', frameon=True, fontsize=11, facecolor='#ffffff', framealpha=0.9, edgecolor='#cbd5e1')
    plt.title('Reliability Diagram: StormCast Uncertainty Calibration', fontsize=14, fontweight='bold', color='#0f172a', pad=20)
    
    plt.tight_layout()
    os.makedirs('../plots', exist_ok=True)
    plt.savefig('reliability_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('../plots/reliability_diagram.png', dpi=300, bbox_inches='tight')
    print("Successfully generated reliability_diagram.png")

if __name__ == "__main__":
    main()
