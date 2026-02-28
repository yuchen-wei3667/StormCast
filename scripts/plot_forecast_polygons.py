#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from StormCast.types import StormState, EnvironmentProfile
from StormCast.diagnostics import compute_storm_core_height, compute_adaptive_steering, compute_effective_shear, compute_bunkers_motion
from StormCast.blending import smooth_observed_motion, calculate_motion_jitter, blend_motion, adjust_weights_for_maturity
from StormCast.kalman import StormKalmanFilter
from StormCast.forecast import forecast_with_uncertainty
from StormCast.config import MIN_VELOCITY_THRESHOLD, MAX_VELOCITY_THRESHOLD, MOTION_SMOOTHING_WINDOW

def main():
    target_file = "/home/yuchenwei/StormCast_Data/20260214/11896.json"
        
    print(f"Using {target_file}")
    with open(target_file, 'r') as f:
        cell_data = json.load(f)
        
    props0 = cell_data[0]['properties']
    wf = props0['wind_field']
    env = EnvironmentProfile(
        winds={850: (wf['u850'], wf['v850']), 700: (wf['u700'], wf['v700']), 
               500: (wf['u500'], wf['v500']), 250: (wf['u250'], wf['v250'])},
        timestamp=datetime.fromisoformat(cell_data[0]['timestamp']),
        freezing_level_km=props0.get('freezing_level_height'),
        mucape=props0.get('MUCAPE')
    )
    
    ref_lat, ref_lon = cell_data[0]['centroid']
    
    def get_poly_meters(bbox):
        poly = []
        if not bbox: return poly
        for lat, lon in bbox:
            y = (lat - ref_lat) * 111111.0
            x = (lon - ref_lon) * 111111.0 * np.cos(np.radians(ref_lat))
            poly.append((x, y))
        return poly

    kf = StormKalmanFilter(initial_state=[0.0, 0.0, 0.0, 0.0])
    motion_history = []
    cur_x, cur_y = 0.0, 0.0
    displacements = []
    
    for step in cell_data:
        cur_x += step.get('dx', 0)
        cur_y += step.get('dy', 0)
        displacements.append((cur_x, cur_y))
        
    # Start forecasting after 10 samples to ensure stable state
    forecast_start_idx = 10
    for i in range(forecast_start_idx + 1):
        step = cell_data[i]
        dx, dy, dt = step.get('dx', 0), step.get('dy', 0), step.get('dt', 0)
        if dt > 0:
            u, v = dx/dt, dy/dt
            motion_history.append((u, v))
            kf.predict(dt=dt)
            kf.update(observation=displacements[i], track_history=len(motion_history))

    step = cell_data[forecast_start_idx]
    h_core = compute_storm_core_height(step['properties'].get('p100EchoTop30'), step['properties'].get('EchoTop50'))
    v_mean = compute_adaptive_steering(env, h_core)
    v_bunkers = compute_bunkers_motion(env, h_core)
    v_obs = smooth_observed_motion(motion_history[-min(len(motion_history), MOTION_SMOOTHING_WINDOW):])
    jitter = calculate_motion_jitter(motion_history[-min(len(motion_history), 10):])
    print(f"Calculated Motion Jitter: {jitter:.3f} m/s")
    
    wf = step['properties']['wind_field']
    shear = np.sqrt((wf['u500'] - wf['u850'])**2 + (wf['v500'] - wf['v850'])**2)
    weights = adjust_weights_for_maturity(h_core, len(motion_history), shear, mucape=env.mucape)
    v_final = blend_motion(v_obs, v_mean, v_bunkers, weights)
    
    poly_coords = get_poly_meters(step.get('bbox', []))
    state = StormState(x=kf.x, y=kf.y, u=v_final[0], v=v_final[1], 
                      h_core=h_core, echo_top_30=step['properties'].get('p100EchoTop30', 10.0), track_history=len(motion_history),
                      motion_jitter=jitter, polygon=poly_coords)
    
    lead_times = [900, 1800, 2700, 3600]
    forecast_points = forecast_with_uncertainty(state, lead_times=lead_times)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot past track
    px = [pt[0]/1000 for pt in displacements[:forecast_start_idx+1]]
    py = [pt[1]/1000 for pt in displacements[:forecast_start_idx+1]]
    ax.plot(px, py, 'k-', alpha=0.3, linewidth=1, label='Past Track')
    
    # Current footprint
    curr_poly_km_x = [pt[0]/1000 for pt in poly_coords]
    curr_poly_km_y = [pt[1]/1000 for pt in poly_coords]
    if curr_poly_km_x:
        curr_poly_km_x.append(curr_poly_km_x[0])
        curr_poly_km_y.append(curr_poly_km_y[0])
        ax.fill(curr_poly_km_x, curr_poly_km_y, color='#374151', alpha=0.6, label='Current Point')

    colors = ['#2563eb', '#d97706', '#16a34a', '#dc2626']
    
    for fp, color in zip(forecast_points, colors):
        if fp.polygon:
            fx = [pt[0]/1000 for pt in fp.polygon]
            fy = [pt[1]/1000 for pt in fp.polygon]
            fx.append(fx[0])
            fy.append(fy[0])
            ax.plot(fx, fy, color=color, linewidth=2, linestyle='-', label=f'{int(fp.lead_time/60)}m Forecast')
            ax.fill(fx, fy, color=color, alpha=0.1)
            
            # Label the centers
            ax.scatter([fp.x/1000], [fp.y/1000], color=color, s=20, zorder=5)

    ax.set_title('StormCast Multi-Lead Polygon Forecast Expansion (Shrunk)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Local X Displacement (km)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Local Y Displacement (km)', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', frameon=True, fontsize=10)
    ax.axis('equal')
    
    save_path = '/home/yuchenwei/.gemini/antigravity/brain/e0405219-20ba-4dbd-ba2d-48d90daf497a/forecast_expansion_sequence.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved expansion sequence plot to {save_path}")

if __name__ == "__main__":
    main()
