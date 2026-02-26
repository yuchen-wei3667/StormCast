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

CHI2_95 = 5.991
LEAD_TIMES = {900: '15 min', 1800: '30 min', 2700: '45 min', 3600: '60 min'}

def evaluate_single_file(file_path, mode="stormcast"):
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
                
                if mode == "bunkers":
                    v_final = compute_bunkers_motion(env, h_core)
                elif mode == "corfidi":
                    v_final = compute_corfidi_motion(env)
                elif mode == "kinematic":
                    v_final = smooth_observed_motion(motion_history[-min(len(motion_history), MOTION_SMOOTHING_WINDOW):])
                else: # stormcast (Production)
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
                            dist_err = np.sqrt((displacements[j][0] - f.x)**2 + (displacements[j][1] - f.y)**2) / 1000.0
                            results.append((lt, dist_err))
                            break
                        if elapsed > lt + 300: break
    return results

def run_benchmark(mode, file_list):
    print(f"Running Benchmark: Mode={mode.upper()} on {len(file_list)} tracks...")
    func = partial(evaluate_single_file, mode=mode)
    with ProcessPoolExecutor() as executor:
        all_results = executor.map(func, file_list)
    
    stats = {lt: [] for lt in LEAD_TIMES.keys()}
    for file_res in all_results:
        for lt, mae in file_res:
            stats[lt].append(mae)
            
    return stats

def main():
    data_dir = "/home/yuchenwei/StormCast_Data"
    file_list = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.json') and f != 'cell_index.json':
                file_list.append(os.path.join(root, f))
    
    modes = ["bunkers", "corfidi", "kinematic", "stormcast"]
    benchmarks = {}
    
    for mode in modes:
        benchmarks[mode] = run_benchmark(mode, file_list)
        
    print("Massive Evaluation complete.")
    
    # Plotting IQR Boxplots
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Add subtle horizontal grid lines behind the boxes
    ax.yaxis.grid(True, linestyle='-', which='major', color='#e5e7eb', alpha=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    
    lt_keys = sorted(LEAD_TIMES.keys())
    mode_labels = ["Pure Bunkers", "Pure Corfidi", "Pure Kinematic", "StormCast Hybrid"]
    
    # Vibrant, distinct categorical palette
    mode_colors = {
        'bunkers': '#94a3b8',   # Slate/Gray
        'corfidi': '#f59e0b',   # Amber/Orange
        'kinematic': '#10b981', # Emerald/Green
        'stormcast': '#4f46e5'  # Indigo/Deep Blue
    }
    
    positions = []
    data_to_plot = []
    colors_to_use = []
    
    group_spacing = 1.8
    for i, lt in enumerate(lt_keys):
        base_pos = i * group_spacing * len(modes)
        for j, mode in enumerate(modes):
            pos = base_pos + j * 1.0
            positions.append(pos)
            data_to_plot.append(benchmarks[mode][lt])
            colors_to_use.append(mode_colors[mode])
            
    bplot = ax.boxplot(data_to_plot, positions=positions, patch_artist=True, widths=0.7, showfliers=False)
    
    for patch, color in zip(bplot['boxes'], colors_to_use):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
        patch.set_edgecolor('#1e293b')
        patch.set_linewidth(1.2)
        
    for median in bplot['medians']:
        median.set_color('#ffffff') # White median line for high contrast
        median.set_linewidth(2.0)
        
    for whisker in bplot['whiskers']:
        whisker.set_color('#475569')
        whisker.set_linewidth(1.5)
        whisker.set_linestyle('-')
        
    for cap in bplot['caps']:
        cap.set_color('#475569')
        cap.set_linewidth(1.5)
        
    # Group names
    ax.set_xticks([i * group_spacing * len(modes) + 1.0 * (len(modes)-1)/2 for i in range(len(lt_keys))])
    ax.set_xticklabels([LEAD_TIMES[lt] for lt in lt_keys], fontsize=13, fontweight='bold', color='#1e293b')
    ax.set_xlabel('Lead Time', fontsize=13, fontweight='bold', color='#1e293b', labelpad=12)
    ax.set_ylabel('Mean Absolute Error (km) - IQR without outliers', fontsize=13, fontweight='bold', color='#1e293b')
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    
    # Custom legend
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=mode_colors[m], label=l) for m, l in zip(modes, mode_labels)]
    ax.legend(handles=legend_patches, loc='upper left', frameon=True, fontsize=12, facecolor='#f8fafc', framealpha=0.95, edgecolor='#cbd5e1', borderpad=0.8)
    
    plt.title('Baseline MAE Interquartile Range Distributions', fontsize=16, fontweight='bold', color='#0f172a', pad=20)
    plt.tight_layout()
    os.makedirs('../plots', exist_ok=True)
    plt.savefig('baseline_iqr.png', dpi=300, bbox_inches='tight')
    plt.savefig('../plots/baseline_iqr.png', dpi=300, bbox_inches='tight')
    print("Successfully restored professional baseline_iqr.png")

if __name__ == "__main__":
    main()
