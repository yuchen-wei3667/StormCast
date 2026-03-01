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
from StormCast.diagnostics import compute_storm_core_height, compute_adaptive_steering, compute_bunkers_motion
from StormCast.blending import smooth_observed_motion, calculate_motion_jitter, blend_motion, adjust_weights_for_maturity
from StormCast.kalman import StormKalmanFilter
from StormCast.forecast import forecast_with_uncertainty
from StormCast.config import MIN_VELOCITY_THRESHOLD, MAX_VELOCITY_THRESHOLD, MOTION_SMOOTHING_WINDOW

def main():
    data_dir = "/home/yuchenwei/StormCast_Data"
    # Find a file with a reasonable track
    target_file = None
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.json') and f != 'cell_index.json':
                full_path = os.path.join(root, f)
                if os.path.getsize(full_path) > 30000: # Need a good track
                    target_file = full_path
                    break
        if target_file: break
        
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
        
    plotted = False
    
    for i, step in enumerate(cell_data):
        if plotted: break
        dx, dy, dt = step.get('dx', 0), step.get('dy', 0), step.get('dt', 0)
        if dt > 0:
            u, v = dx/dt, dy/dt
            motion_history.append((u, v))
            kf.predict(dt=dt)
            kf.update(observation=displacements[i], track_history=len(motion_history))
            
            hist_len = len(motion_history)
            if hist_len == 5: # Let it build some history
                h_core = compute_storm_core_height(step['properties'].get('p100EchoTop30'), step['properties'].get('EchoTop50'))
                v_mean = compute_adaptive_steering(env, h_core)
                v_bunkers = compute_bunkers_motion(env, h_core)
                
                v_obs = smooth_observed_motion(motion_history[-min(len(motion_history), MOTION_SMOOTHING_WINDOW):])
                jitter = calculate_motion_jitter(motion_history[-min(len(motion_history), 10):])
                
                wf = step['properties']['wind_field']
                shear = np.sqrt((wf['u500'] - wf['u850'])**2 + (wf['v500'] - wf['v850'])**2)
                weights = adjust_weights_for_maturity(h_core, hist_len, shear, mucape=env.mucape)
                v_final = blend_motion(v_obs, v_mean, v_bunkers, weights)
                
                poly_coords = get_poly_meters(step.get('bbox', []))
                
                state = StormState(x=kf.x, y=kf.y, u=v_final[0], v=v_final[1], 
                                  h_core=h_core, echo_top_30=step['properties'].get('p100EchoTop30', 10.0), track_history=hist_len,
                                  motion_jitter=jitter, polygon=poly_coords)
                
                # Find a step approx 10+ min later
                for j in range(i + 1, len(cell_data)):
                    elapsed = sum(cell_data[k].get('dt', 0) for k in range(i + 1, j + 1))
                    if elapsed > 600:
                        f = forecast_with_uncertainty(state, lead_times=[elapsed])[0]
                        actual_bbox = cell_data[j].get('bbox', [])
                        if actual_bbox and f.polygon:
                            actual_poly_pts = get_poly_meters(actual_bbox)
                            actual_poly = Polygon(actual_poly_pts)
                            fcst_poly = Polygon(f.polygon)
                            
                            # Plotting
                            fig, ax = plt.subplots(figsize=(8, 8))
                            
                            # Plot track up to now
                            px = [pt[0]/1000 for pt in displacements[:i+1]]
                            py = [pt[1]/1000 for pt in displacements[:i+1]]
                            ax.plot(px, py, 'k--', alpha=0.5, label='Past Track')
                            ax.scatter([px[-1]], [py[-1]], color='black', marker='o', s=50, label='Current Point')
                            
                            # Plot true future center
                            ax.scatter([displacements[j][0]/1000], [displacements[j][1]/1000], color='blue', marker='x', s=100, label=f'True Future ({elapsed//60}m)')
                            
                            # Plot current footprint
                            curr_poly_km_x = [pt[0]/1000 for pt in poly_coords]
                            curr_poly_km_y = [pt[1]/1000 for pt in poly_coords]
                            curr_poly_km_x.append(curr_poly_km_x[0])
                            curr_poly_km_y.append(curr_poly_km_y[0])
                            ax.plot(curr_poly_km_x, curr_poly_km_y, 'k-', linewidth=2, label='Current Footprint')
                            
                            # Plot actual future footprint
                            act_poly_km_x = [pt[0]/1000 for pt in actual_poly_pts]
                            act_poly_km_y = [pt[1]/1000 for pt in actual_poly_pts]
                            act_poly_km_x.append(act_poly_km_x[0])
                            act_poly_km_y.append(act_poly_km_y[0])
                            ax.fill(act_poly_km_x, act_poly_km_y, color='blue', alpha=0.3, label=f'True Footprint ({elapsed//60}m)')
                            
                            # Plot forecasted footprint
                            fcst_poly_km_x = [pt[0]/1000 for pt in f.polygon]
                            fcst_poly_km_y = [pt[1]/1000 for pt in f.polygon]
                            fcst_poly_km_x.append(fcst_poly_km_x[0])
                            fcst_poly_km_y.append(fcst_poly_km_y[0])
                            ax.plot(fcst_poly_km_x, fcst_poly_km_y, 'r--', linewidth=2, label='Forecast Buffer (Tuned-90%)')
                            ax.fill(fcst_poly_km_x, fcst_poly_km_y, color='red', alpha=0.1)
                            
                            # Plot 0-30m Encompassing Polygon
                            if hasattr(f, 'polygon_0_30m') and f.polygon_0_30m:
                                poly_0_30_x = [pt[0]/1000 for pt in f.polygon_0_30m]
                                poly_0_30_y = [pt[1]/1000 for pt in f.polygon_0_30m]
                                poly_0_30_x.append(poly_0_30_x[0])
                                poly_0_30_y.append(poly_0_30_y[0])
                                ax.plot(poly_0_30_x, poly_0_30_y, color='purple', linewidth=2, linestyle='-.', label='0-30m Encompassing Polygon')
                                ax.fill(poly_0_30_x, poly_0_30_y, color='purple', alpha=0.1)
                            
                            # Calculate overlap
                            overlap = 0
                            if actual_poly.is_valid and fcst_poly.is_valid:
                                overlap_area = actual_poly.intersection(fcst_poly).area
                                overlap = (overlap_area / actual_poly.area) * 100
                            
                            ax.set_title(f'{elapsed//60} Minute Polygon Forecast Projection\nOverlap: {overlap:.1f}%')
                            ax.set_xlabel('Local X Displacement (km)')
                            ax.set_ylabel('Local Y Displacement (km)')
                            ax.grid(True, linestyle=':', alpha=0.6)
                            ax.legend(loc='upper left')
                            ax.axis('equal')
                            
                            save_path = '/home/yuchenwei/.gemini/antigravity/brain/e0405219-20ba-4dbd-ba2d-48d90daf497a/sample_polygon.png'
                            plt.savefig(save_path, dpi=300, bbox_inches='tight')
                            print(f"Saved plot to {save_path}")
                            plotted = True
                            
                        break

if __name__ == "__main__":
    main()
