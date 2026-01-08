#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
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
from StormCast.uncertainty import generate_uncertainty_ellipse

def latlon_to_relative_meters(bbox, centroid, cum_x, cum_y):
    """Convert bbox [lat, lon] to relative meters from plot origin."""
    ref_lat, ref_lon = centroid
    rel_coords = []
    for lat, lon in bbox:
        dy = (lat - ref_lat) * 111111.0
        dx = (lon - ref_lon) * 111111.0 * np.cos(np.radians(ref_lat))
        rel_coords.append((cum_x + dx, cum_y + dy))
    return rel_coords

def plot_storm_on_ax(ax, file_path, forecast_step=30):
    with open(file_path, 'r') as f:
        cell_data = json.load(f)
    
    if len(cell_data) < forecast_step + 5:
        return False

    past_x, past_y = [], []
    future_x, future_y = [], []
    cum_x, cum_y = 0.0, 0.0
    
    # Environment
    p0 = cell_data[0]['properties']
    env = EnvironmentProfile(
        winds={850: (p0['u850'], p0['v850']), 700: (p0['u700'], p0['v700']), 
               500: (p0['u500'], p0['v500']), 250: (p0['u250'], p0['v250'])},
        timestamp=datetime.fromisoformat(cell_data[0]['timestamp'])
    )
    
    kf = StormKalmanFilter(initial_state=[0.0, 0.0, 0.0, 0.0])
    motion_history = []
    
    for i in range(forecast_step + 1):
        step = cell_data[i]
        dx, dy, dt = step.get('dx', 0), step.get('dy', 0), step.get('dt', 0)
        cum_x += dx
        cum_y += dy
        past_x.append(cum_x)
        past_y.append(cum_y)
        if dt > 0:
            motion_history.append((dx/dt, dy/dt))
            kf.predict(dt=dt)
            kf.update(observation=(cum_x, cum_y), track_history=len(motion_history))
            
    # Save current polygon
    curr_step = cell_data[forecast_step]
    current_poly = latlon_to_relative_meters(curr_step['bbox'], curr_step['centroid'], kf.x, kf.y)
            
    # Future track (capping at 60 minutes / 3600s for direct comparison)
    future_time_elapsed = 0
    actual_positions = {}
    target_leads = [900, 1800, 2700, 3600]
    
    for i in range(forecast_step + 1, len(cell_data)):
        dt = cell_data[i].get('dt', 0)
        future_time_elapsed += dt
        cum_x += cell_data[i].get('dx', 0)
        cum_y += cell_data[i].get('dy', 0)
        future_x.append(cum_x)
        future_y.append(cum_y)
        
        # Save actual future polygons at lead times
        for lead in target_leads:
            if lead not in actual_positions and abs(future_time_elapsed - lead) < 150:
                actual_positions[lead] = {
                    'pos': (cum_x, cum_y),
                    'poly': latlon_to_relative_meters(cell_data[i]['bbox'], cell_data[i]['centroid'], cum_x, cum_y)
                }
        
        if future_time_elapsed >= 3600:
            break
            
    # Forecast with uncertainty
    curr = cell_data[forecast_step]
    h_core = compute_storm_core_height(curr['properties']['EchoTop30'], curr['properties']['EchoTop50'])
    v_mean = compute_adaptive_steering(env, h_core)
    v_bunkers = compute_bunkers_motion(env, h_core)
    v_obs = smooth_observed_motion(motion_history[-5:])
    
    # Calculate motion jitter
    jitter = calculate_motion_jitter(motion_history[-10:])
    
    props = curr['properties']
    shear = np.sqrt((props['u500'] - props['u850'])**2 + (props['v500'] - props['v850'])**2)
    weights = adjust_weights_for_maturity(h_core, len(motion_history), shear)
    v_final = blend_motion(v_obs, v_mean, v_bunkers, weights)
    
    state = StormState(x=kf.x, y=kf.y, u=v_final[0], v=v_final[1], 
                      h_core=h_core, track_history=len(motion_history),
                      motion_jitter=jitter)
    
    forecast_points = forecast_with_uncertainty(state, lead_times=target_leads)
    
    # Plotting
    ax.set_facecolor('#1a1a1a')
    ax.plot(np.array(past_x)/1000, np.array(past_y)/1000, color='cyan', linewidth=1.5, label='Past')
    ax.plot(np.array(future_x)/1000, np.array(future_y)/1000, color='white', linestyle='--', alpha=0.6, label='Actual')
    ax.scatter(kf.x/1000, kf.y/1000, color='yellow', s=30, zorder=5)
    
    # Plot Current Polygon
    px, py = zip(*current_poly)
    ax.fill(np.array(px)/1000, np.array(py)/1000, color='yellow', alpha=0.2, hatch='///', label='Observed Cell')
    ax.plot(np.array(px)/1000, np.array(py)/1000, color='yellow', linewidth=0.8, alpha=0.5)
    
    colors = ['#FFD700', '#FFA500', '#FF8C00', '#FF4500']
    for i, fp in enumerate(forecast_points):
        ellipse = generate_uncertainty_ellipse(fp.sigma_x, fp.sigma_y)
        ex = [(fp.x + e[0])/1000 for e in ellipse]
        ey = [(fp.y + e[1])/1000 for e in ellipse]
        ex.append(ex[0]) # Close loop
        ey.append(ey[0])
        ax.fill(ex, ey, color=colors[i], alpha=0.15)
        ax.plot(ex, ey, color=colors[i], alpha=0.4, linewidth=0.8)
        
        # Plot Forecast Marker
        ax.scatter(fp.x/1000, fp.y/1000, color=colors[i], marker='x', s=20)
        
        # Plot GROUND TRUTH Marker and Polygon if available
        lead_val = int(fp.lead_time)
        if lead_val in actual_positions:
            act_data = actual_positions[lead_val]
            ax_x, ax_y = act_data['pos']
            ax.scatter(ax_x/1000, ax_y/1000, color='white', marker='o', s=15, edgecolors='black', linewidths=0.5, alpha=0.8, zorder=6)
            
            # Actual Poly outline
            apx, apy = zip(*act_data['poly'])
            ax.plot(np.array(apx)/1000, np.array(apy)/1000, color='white', linewidth=1.0, alpha=0.6, linestyle=':')
            
    ax.set_title(f"Storm {os.path.basename(file_path).split('.')[0]} (J: {jitter:.1f})", color='white', fontsize=10)
    ax.tick_params(colors='gray', labelsize=8)
    ax.grid(True, alpha=0.1, color='gray')
    ax.set_aspect('equal')
    return True

def create_panel_for_date(date_dir, output_path):
    files = [f for f in os.listdir(date_dir) if f.endswith('.json') and f != 'cell_index.json']
    candidate_files = []
    for f in files:
        if len(candidate_files) >= 20: break
        path = os.path.join(date_dir, f)
        with open(path, 'r') as jf:
            data = json.load(jf)
            if len(data) >= 50: candidate_files.append(path)
    
    if len(candidate_files) < 8: candidate_files = [os.path.join(date_dir, f) for f in files[:8]]

    fig, axes = plt.subplots(4, 2, figsize=(12, 18), dpi=120)
    fig.patch.set_facecolor('#0f0f0f')
    plt.subplots_adjust(hspace=0.3, wspace=0.2, top=0.92, bottom=0.05)
    
    date_str = os.path.basename(date_dir)
    fig.suptitle(f"StormCast Dynamic Uncertainty - {date_str}", color='white', fontsize=20, fontweight='bold')
    
    count = 0
    for ax in axes.flatten():
        if count < len(candidate_files):
            plot_storm_on_ax(ax, candidate_files[count], forecast_step=20)
            count += 1
        else:
            ax.axis('off')
            
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

def main():
    data_path = "/home/yuchenwei/StormCast/StormCast_Training_Data"
    out_dir = "/home/yuchenwei/Projects/StormCast/plots"
    os.makedirs(out_dir, exist_ok=True)
    dates = [d for d in sorted(os.listdir(data_path)) if os.path.isdir(os.path.join(data_path, d))]
    for date in dates[-3:]:
        create_panel_for_date(os.path.join(data_path, date), os.path.join(out_dir, f"panel_{date}.png"))

if __name__ == "__main__":
    main()
