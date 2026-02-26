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
    compute_bunkers_motion,
    compute_raw_bunkers_motion,
    compute_corfidi_motion
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

def get_hcore_bin(h):
    if h < 4.0:
        return "0-4km"
    elif h < 6.0:
        return "4-6km"
    elif h < 8.0:
        return "6-8km"
    elif h < 10.0:
        return "8-10km"
    else:
        return "10km+"

def evaluate_single_file(file_path, mode="stormcast"):
    results = [] # list of (hcore_bin, mae)
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
                
                if mode == "bunkers":
                    v_final = compute_raw_bunkers_motion(env)
                elif mode == "corfidi":
                    v_final = compute_corfidi_motion(env)
                elif mode == "kinematic":
                    v_final = smooth_observed_motion(motion_history[-min(len(motion_history), MOTION_SMOOTHING_WINDOW):])
                else: # stormcast
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
                
                # ONLY Evaluating 30 min (1800s)
                target_lt = 1800
                elapsed = 0
                for j in range(i + 1, len(cell_data)):
                    elapsed += cell_data[j].get('dt', 0)
                    if abs(elapsed - target_lt) <= 150:
                        f = forecast_with_uncertainty(state, lead_times=[elapsed])[0]
                        dist_err = np.sqrt((displacements[j][0] - f.x)**2 + (displacements[j][1] - f.y)**2) / 1000.0
                        hbin = get_hcore_bin(h_core)
                        results.append((hbin, dist_err))
                        break
                    if elapsed > target_lt + 300: break
    return results

def run_benchmark(mode, file_list, all_bins):
    print(f"Running Benchmark: Mode={mode.upper()} on {len(file_list)} tracks...")
    func = partial(evaluate_single_file, mode=mode)
    with ProcessPoolExecutor() as executor:
        all_results = executor.map(func, file_list)
    
    stats = {b: [] for b in all_bins}
    for file_res in all_results:
        for hbin, mae in file_res:
            stats[hbin].append(mae)
            
    return stats

def main():
    data_dir = "/home/yuchenwei/StormCast_Data"
    file_list = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.json') and f != 'cell_index.json':
                file_list.append(os.path.join(root, f))
    
    modes = ["bunkers", "corfidi", "kinematic", "stormcast"]
    bins = ["0-4km", "4-6km", "6-8km", "8-10km", "10km+"]
    benchmarks = {}
    
    for mode in modes:
        benchmarks[mode] = run_benchmark(mode, file_list, bins)
        
    print("Evaluation complete. Plotting...")
    
    # Calculate means, SEMs and counts
    means = {m: [] for m in modes}
    sems = {m: [] for m in modes}
    counts = {b: 0 for b in bins}
    
    for b in bins:
        counts[b] = len(benchmarks["stormcast"][b]) # Count is same for all modes
        for m in modes:
            arr = np.array(benchmarks[m][b])
            if len(arr) > 0:
                means[m].append(np.mean(arr))
                sems[m].append(np.std(arr) / np.sqrt(len(arr)))
            else:
                means[m].append(0)
                sems[m].append(0)

    # Plotting Grouped Bar Chart
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    ax1.yaxis.grid(True, linestyle='-', which='major', color='#e5e7eb', alpha=0.7)
    ax1.xaxis.grid(False)
    ax1.set_axisbelow(True)
    
    mode_labels = ["Pure Bunkers", "Pure Corfidi", "Pure Kinematic", "StormCast Hybrid"]
    mode_colors = {
        'bunkers': '#94a3b8',   # Slate/Gray
        'corfidi': '#f59e0b',   # Amber/Orange
        'kinematic': '#10b981', # Emerald/Green
        'stormcast': '#4f46e5'  # Indigo/Deep Blue
    }
    
    x = np.arange(len(bins))
    width = 0.2
    
    # Plot bars with error brackets (SEM)
    error_kw = dict(lw=1.5, capsize=4, capthick=1.5, ecolor='#1e293b')
    
    ax1.bar(x - 1.5*width, means['bunkers'], width, yerr=sems['bunkers'], label='Pure Bunkers', 
            color=mode_colors['bunkers'], edgecolor='#1e293b', linewidth=1.0, alpha=0.9, error_kw=error_kw)
    ax1.bar(x - 0.5*width, means['corfidi'], width, yerr=sems['corfidi'], label='Pure Corfidi', 
            color=mode_colors['corfidi'], edgecolor='#1e293b', linewidth=1.0, alpha=0.9, error_kw=error_kw)
    ax1.bar(x + 0.5*width, means['kinematic'], width, yerr=sems['kinematic'], label='Pure Kinematic', 
            color=mode_colors['kinematic'], edgecolor='#1e293b', linewidth=1.0, alpha=0.9, error_kw=error_kw)
    ax1.bar(x + 1.5*width, means['stormcast'], width, yerr=sems['stormcast'], label='StormCast Hybrid', 
            color=mode_colors['stormcast'], edgecolor='#1e293b', linewidth=1.0, alpha=0.9, error_kw=error_kw)
    
    # Adding 'n=[count]' labels at the top of each bin group
    for i, b in enumerate(bins):
        # find the highest point in this group (bar + SEM) to place the label above it
        group_max = max([means[m][i] + sems[m][i] for m in modes])
        ax1.text(i, group_max + 0.1, f"n={counts[b]}", ha='center', va='bottom', 
                 fontsize=10, fontweight='bold', color='#64748b')
    
    # Styling
    ax1.set_xlabel('Radar-derived Storm Core Height (Hcore)', fontsize=13, fontweight='bold', color='#1e293b', labelpad=12)
    ax1.set_ylabel('30-Minute Mean Absolute Error (km)', fontsize=13, fontweight='bold', color='#1e293b')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bins, fontsize=12, fontweight='bold', color='#1e293b')
    
    # Adjust ymax for better visual balance and space for labels
    max_mae = max([max(means[m]) for m in modes])
    ax1.set_ylim(0, max_mae * 1.35) # Increased buffer for top labels
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#cbd5e1')
    ax1.spines['bottom'].set_color('#cbd5e1')
    
    # Legends
    lines, labels = ax1.get_legend_handles_labels()
    # Moved to 'upper left' to ensure 10km+ bars on right are visible
    ax1.legend(lines, labels, loc='upper left', frameon=True, fontsize=11, facecolor='#ffffff', framealpha=0.95, edgecolor='#cbd5e1')
    
    plt.title('30-Min Forecast Error vs. Storm Vertical Structure', fontsize=16, fontweight='bold', color='#0f172a', pad=20)
    plt.tight_layout()
    os.makedirs('../plots', exist_ok=True)
    plt.savefig('mae_vs_hcore.png', dpi=300, bbox_inches='tight')
    plt.savefig('../plots/mae_vs_hcore.png', dpi=300, bbox_inches='tight')
    print("Successfully generated mae_vs_hcore.png")

if __name__ == "__main__":
    main()
