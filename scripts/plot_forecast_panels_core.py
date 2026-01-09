#!/home/yuchenwei/.conda/envs/StormCast/bin/python3
"""
StormCast Panel Generator (Core Engine version)

Regenerates the summary panels for each date using the StormCastEngine integration
instead of manual component-by-component logic.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from StormCast import StormCastEngine, EnvironmentProfile
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
    
    # 1. Initialize Engine
    # Use first centroid as reference
    p0 = cell_data[0]
    ref_lat, ref_lon = p0['centroid']
    engine = StormCastEngine(reference_lat=ref_lat, reference_lon=ref_lon)
    
    # 2. Setup Environment
    props0 = p0['properties']
    env = EnvironmentProfile(
        winds={
            850: (props0['u850'], props0['v850']), 
            700: (props0['u700'], props0['v700']), 
            500: (props0['u500'], props0['v500']), 
            250: (props0['u250'], props0['v250'])
        },
        timestamp=datetime.fromisoformat(p0['timestamp'])
    )
    engine.set_environment(env)
    
    # 3. Feed History
    for i in range(forecast_step + 1):
        step = cell_data[i]
        dx, dy, dt = step.get('dx', 0), step.get('dy', 0), step.get('dt', 0)
        cum_x += dx
        cum_y += dy
        past_x.append(cum_x)
        past_y.append(cum_y)
        
        # Add to engine
        engine.add_observation(
            x=cum_x, 
            y=cum_y, 
            dt_seconds=dt, 
            echo_top_30=step['properties']['EchoTop30'],
            echo_top_50=step['properties']['EchoTop50'],
            timestamp=datetime.fromisoformat(step['timestamp'])
        )
            
    # Save current polygon coordinates for plotting
    curr_step = cell_data[forecast_step]
    current_poly = latlon_to_relative_meters(curr_step['bbox'], curr_step['centroid'], engine.kalman_filter.x, engine.kalman_filter.y)
            
    # 4. Extract Ground Truth for Comparison
    future_time_elapsed = 0
    actual_positions = {}
    target_leads = [900, 1800, 2700, 3600]
    
    # Track future actual points (not seen by engine)
    temp_x, temp_y = cum_x, cum_y
    for i in range(forecast_step + 1, len(cell_data)):
        dt = cell_data[i].get('dt', 0)
        future_time_elapsed += dt
        temp_x += cell_data[i].get('dx', 0)
        temp_y += cell_data[i].get('dy', 0)
        future_x.append(temp_x)
        future_y.append(temp_y)
        
        # Save actual future polygons at lead times
        for lead in target_leads:
            if lead not in actual_positions and abs(future_time_elapsed - lead) < 150:
                actual_positions[lead] = {
                    'pos': (temp_x, temp_y),
                    'poly': latlon_to_relative_meters(cell_data[i]['bbox'], cell_data[i]['centroid'], temp_x, temp_y)
                }
        
        if future_time_elapsed >= 3600:
            break
            
    # 5. Generate Forecast with Core Engine
    result = engine.generate_forecast(lead_times=target_leads, confidence=0.95)
    
    # 6. Plotting
    ax.set_facecolor('#1a1a1a')
    ax.plot(np.array(past_x)/1000, np.array(past_y)/1000, color='cyan', linewidth=1.5, label='Past')
    ax.plot(np.array(future_x)/1000, np.array(future_y)/1000, color='white', linestyle='--', alpha=0.6, label='Actual')
    
    # Current Position
    ax.scatter(engine.kalman_filter.x/1000, engine.kalman_filter.y/1000, color='yellow', s=30, zorder=5)
    
    # Current Polygon
    px, py = zip(*current_poly)
    ax.fill(np.array(px)/1000, np.array(py)/1000, color='yellow', alpha=0.2, hatch='///', label='Observed Cell')
    ax.plot(np.array(px)/1000, np.array(py)/1000, color='yellow', linewidth=0.8, alpha=0.5)
    
    colors = ['#FFD700', '#FFA500', '#FF8C00', '#FF4500']
    
    # Re-extract standard sigmas for ellipses (plotting in core uses Circles, 
    # but here we want the old ellipse look if we can, or just circles. 
    # I'll use ellipses to match the aesthetic of plot_forecast_panels.py).
    # Since ForecastResult doesn't give sigmas, I'll calculate them locally 
    # or just use the radius. I'll use the radius as a circle for consistency 
    # with the user's recent request for "center, radius".
    
    for i, cone in enumerate(result.forecast_cones):
        cx = cone['x']
        cy = cone['y']
        radius = cone['radius']
        
        # Plot as circle
        circle = plt.Circle((cx/1000, cy/1000), radius/1000, color=colors[i], alpha=0.15)
        ax.add_patch(circle)
        ax.plot([], []) # Dummy
        
        # Outline
        circle_out = plt.Circle((cx/1000, cy/1000), radius/1000, color=colors[i], 
                               fill=False, alpha=0.4, linewidth=0.8)
        ax.add_patch(circle_out)
        
        # Forecast Marker
        ax.scatter(cx/1000, cy/1000, color=colors[i], marker='x', s=20)
        
        # Ground Truth comparison
        lead_val = int(cone['lead_time'])
        # Find closest ground truth
        matching_gt = None
        for k in actual_positions:
            if abs(k - lead_val) < 150:
                matching_gt = actual_positions[k]
                break
                
        if matching_gt:
            ax_x, ax_y = matching_gt['pos']
            ax.scatter(ax_x/1000, ax_y/1000, color='white', marker='o', s=15, 
                       edgecolors='black', linewidths=0.5, alpha=0.8, zorder=6)
            
            # Actual Poly outline
            apx, apy = zip(*matching_gt['poly'])
            ax.plot(np.array(apx)/1000, np.array(apy)/1000, color='white', linewidth=1.0, 
                    alpha=0.6, linestyle=':')
            
    ax.set_title(f"Cell {os.path.basename(file_path).split('.')[0]}", color='white', fontsize=10)
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
    fig.suptitle(f"StormCast Core Integrated - {date_str}", color='white', fontsize=20, fontweight='bold')
    
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
    for date in dates:
        print(f"Processing date: {date}")
        output_file = os.path.join(out_dir, f"panel_core_{date}.png")
        create_panel_for_date(os.path.join(data_path, date), output_file)
        print(f"  Saved to {output_file}")

if __name__ == "__main__":
    main()
