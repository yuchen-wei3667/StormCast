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

LEAD_TIMES = {900: '15 min', 1800: '30 min', 2700: '45 min', 3600: '60 min'}
MODES = ["bunkers", "corfidi", "kinematic", "stormcast"]

def evaluate_single_file(file_path):
    results = [] # list of (mode, lt, along_err, cross_err)
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
                v_bunkers_guidance = compute_bunkers_motion(env, h_core)
                v_corfidi_guidance = compute_corfidi_motion(env, h_core)
                v_obs = smooth_observed_motion(motion_history[-min(len(motion_history), MOTION_SMOOTHING_WINDOW):])
                
                wf_step = step['properties']['wind_field']
                shear = np.sqrt((wf_step['u500'] - wf_step['u850'])**2 + (wf_step['v500'] - wf_step['v850'])**2)
                weights = adjust_weights_for_maturity(h_core, hist_len, shear, mucape=env.mucape)
                
                jitter = calculate_motion_jitter(motion_history[-min(len(motion_history), 10):])
                
                # Setup states for all modes
                # 1. StormCast
                v_sc = blend_motion(v_obs, v_mean, v_bunkers_guidance, weights)
                sc_state = StormState(x=kf.x, y=kf.y, u=v_sc[0], v=v_sc[1], h_core=h_core, track_history=hist_len, motion_jitter=jitter)
                
                # 2. Bunkers (Pure)
                v_bnk = v_bunkers_guidance
                bnk_state = StormState(x=displacements[i][0], y=displacements[i][1], u=v_bnk[0], v=v_bnk[1], h_core=h_core)
                
                # 3. Kinematic (Pure)
                kin_state = StormState(x=displacements[i][0], y=displacements[i][1], u=v_obs[0], v=v_obs[1], h_core=h_core)
                
                # 4. Corfidi (Pure)
                v_cor = v_corfidi_guidance
                cor_state = StormState(x=displacements[i][0], y=displacements[i][1], u=v_cor[0], v=v_cor[1], h_core=h_core)
                
                mode_states = {"stormcast": sc_state, "bunkers": bnk_state, "kinematic": kin_state, "corfidi": cor_state}
                
                for lt in LEAD_TIMES.keys():
                    elapsed = 0
                    for j in range(i + 1, len(cell_data)):
                        elapsed += cell_data[j].get('dt', 0)
                        if abs(elapsed - lt) <= 150:
                            # Target position
                            tx, ty = displacements[j]
                            sx, sy = displacements[i]
                            
                            # Actual translation vector
                            dx_truth, dy_truth = tx - sx, ty - sy
                            dist_truth = np.sqrt(dx_truth**2 + dy_truth**2)
                            
                            if dist_truth < 1.0: # Avoid division by zero
                                break
                                
                            # Along-track unit vector
                            u_along, v_along = dx_truth / dist_truth, dy_truth / dist_truth
                            # Cross-track unit vector (90 deg CCW)
                            u_cross, v_cross = -v_along, u_along
                            
                            for mode, state in mode_states.items():
                                f = forecast_with_uncertainty(state, lead_times=[elapsed])[0]
                                
                                # Error vector (Forecast - Truth)
                                ex, ey = f.x - tx, f.y - ty
                                
                                # Decompose
                                # dot product: (ex * u_along + ey * v_along)
                                along_err = abs(ex * u_along + ey * v_along) / 1000.0
                                # dot product: (ex * u_cross + ey * v_cross)
                                cross_err = abs(ex * u_cross + ey * v_cross) / 1000.0
                                
                                results.append((mode, lt, along_err, cross_err))
                            break
                        if elapsed > lt + 300: break
    return results

def main():
    data_dir = "/home/yuchenwei/StormCast_Data"
    file_list = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.json') and f != 'cell_index.json':
                file_list.append(os.path.join(root, f))
                
    print(f"Beginning Error Decomposition on {len(file_list)} tracks...")
    
    with ProcessPoolExecutor() as executor:
        all_results = list(executor.map(evaluate_single_file, file_list))
        
    # Aggregate
    agg = {m: {lt: {"along": [], "cross": []} for lt in LEAD_TIMES.keys()} for m in MODES}
    
    for file_res in all_results:
        for mode, lt, along, cross in file_res:
            agg[mode][lt]["along"].append(along)
            agg[mode][lt]["cross"].append(cross)
            
    # Calculate means and SEMs
    final_stats = {m: {lt: {"along": 0.0, "cross": 0.0, "along_sem": 0.0, "cross_sem": 0.0} for lt in LEAD_TIMES.keys()} for m in MODES}
    for m in MODES:
        for lt in LEAD_TIMES.keys():
            if agg[m][lt]["along"]:
                arr_along = np.array(agg[m][lt]["along"])
                arr_cross = np.array(agg[m][lt]["cross"])
                final_stats[m][lt]["along"] = np.mean(arr_along)
                final_stats[m][lt]["cross"] = np.mean(arr_cross)
                final_stats[m][lt]["along_sem"] = np.std(arr_along) / np.sqrt(len(arr_along))
                final_stats[m][lt]["cross_sem"] = np.std(arr_cross) / np.sqrt(len(arr_cross))

    print("Success. Plotting side-by-side IQR + Mean decomposition...")

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    
    lt_keys = sorted(LEAD_TIMES.keys())
    lt_labels = [LEAD_TIMES[k] for k in lt_keys]
    group_spacing = 2.5
    
    mode_labels_map = {
        "bunkers": "Pure Bunkers",
        "corfidi": "Pure Corfidi",
        "kinematic": "Pure Kinematic",
        "stormcast": "StormCast Hybrid"
    }
    
    colors_map = {
        "bunkers": "#94a3b8",   # Slate
        "corfidi": "#f59e0b",   # Amber
        "kinematic": "#10b981", # Emerald
        "stormcast": "#4f46e5"  # Indigo
    }

    def plot_iqr_mean_group(ax, data_key, title):
        ax.yaxis.grid(True, linestyle='-', which='major', color='#f1f5f9', alpha=0.8)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        
        positions = []
        data_to_plot = []
        colors_to_use = []
        means = []
        
        for i, lt in enumerate(lt_keys):
            base_pos = i * group_spacing * len(MODES)
            for j, mode in enumerate(MODES):
                pos = base_pos + j * 1.0
                positions.append(pos)
                arr = agg[mode][lt][data_key]
                data_to_plot.append(arr)
                colors_to_use.append(colors_map[mode])
                means.append(np.mean(arr) if len(arr) > 0 else 0)
                
        # Plot Boxplot (IQR)
        bplot = ax.boxplot(data_to_plot, positions=positions, patch_artist=True, widths=0.6, 
                           showfliers=False, medianprops=dict(color='white', linewidth=1.5),
                           whiskerprops=dict(color='#64748b', linewidth=1.2),
                           capprops=dict(color='#64748b', linewidth=1.2))
        
        for patch, color in zip(bplot['boxes'], colors_to_use):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('#1e293b')
            patch.set_linewidth(1.0)
            
        # Overlay Mean markers
        ax.scatter(positions, means, marker='D', s=30, color='#ef4444', label='Mean MAE', zorder=5, edgecolor='black', linewidth=0.5)
            
        ax.set_xticks([i * group_spacing * len(MODES) + 1.0 * (len(MODES)-1)/2 for i in range(len(lt_keys))])
        ax.set_xticklabels(lt_labels, fontsize=12, fontweight='bold', color='#1e293b')
        ax.set_title(title, fontsize=15, fontweight='bold', color='#0f172a', pad=15)
        ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')

    plot_iqr_mean_group(ax1, "along", "Along-Track Error (Speed Bias)")
    plot_iqr_mean_group(ax2, "cross", "Cross-Track Error (Directional Bias)")
    
    ax1.set_ylabel('Absolute Error (km)', fontsize=13, fontweight='bold', color='#1e293b')
    
    # Shared Legend at the bottom
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=colors_map[m], label=mode_labels_map[m]) for m in MODES]
    legend_patches.append(plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#ef4444', markersize=8, label='Mean MAE', markeredgecolor='black'))
    
    fig.legend(handles=legend_patches, loc='lower center', ncol=5, frameon=True, 
               fontsize=11, facecolor='#ffffff', edgecolor='#cbd5e1', bbox_to_anchor=(0.5, -0.05))

    plt.suptitle('StormCast Error Decomposition (IQR + Mean)', fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    os.makedirs('../plots', exist_ok=True)
    plt.savefig('../plots/error_decomposition.png', dpi=300, bbox_inches='tight')
    print("Successfully generated professional error_decomposition.png (IQR + Mean)")

if __name__ == "__main__":
    main()
