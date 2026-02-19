#!/usr/bin/env python3
"""
StormCast Real Data Evaluation Script

Runs the StormCast framework on actual storm cell JSON files and reports any errors,
numerical instabilities, and forecast accuracy metrics (MAE and Error Range Coverage).
"""

import os
import json
import numpy as np
import sys
import glob
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
    blend_motion,
    adjust_weights_for_maturity
)
from StormCast.kalman import StormKalmanFilter
from StormCast.forecast import forecast_with_uncertainty

def process_file(file_path, overrides=None):
    """
    Process a single storm cell JSON file with multi-interval evaluation.
    """
    with open(file_path, 'r') as f:
        cell_data = json.load(f)
    
    if not isinstance(cell_data, list) or len(cell_data) < 2:
        return None

    results = {
        'errors': [],
        'steps_tracked': 0
    }
    
    o = overrides or {}
    window = o.get('window', 5)
    
    # Initialize
    first_step = cell_data[0]
    props = first_step['properties']
    wind_data = props.get('wind_field', props)
    
    from StormCast.config import PRESSURE_LEVELS
    
    winds = {}
    for p in PRESSURE_LEVELS:
        # Some datasets might be missing certain levels, use 0 or skip
        u = wind_data.get(f'u{p}')
        v = wind_data.get(f'v{p}')
        if u is not None and v is not None:
            winds[p] = (u, v)

    # Prepare environment
    env = EnvironmentProfile(
        winds=winds,
        timestamp=datetime.fromisoformat(first_step['timestamp'])
    )
    
    kf = StormKalmanFilter(initial_state=[0.0, 0.0, 0.0, 0.0])
    
    motion_history = []
    cumulative_x = 0.0
    cumulative_y = 0.0
    
    CHI2_95 = 5.991
    
    for i, step in enumerate(cell_data):
        props = step['properties']
        dx = step.get('dx')
        dy = step.get('dy')
        dt = step.get('dt')
        
        if dx is not None: cumulative_x += dx
        if dy is not None: cumulative_y += dy
        
        if dx is None or dy is None or dt is None or dt == 0:
            continue
            
        u_obs_raw = dx / dt
        v_obs_raw = dy / dt
        motion_history.append((u_obs_raw, v_obs_raw))
        
        try:
            # Note: Using p100EchoTop30 based on data analysis
            h_core = compute_storm_core_height(props.get('p100EchoTop30'), props.get('EchoTop50'))
            v_mean_star = compute_adaptive_steering(env, h_core)
            v_bunkers = compute_bunkers_motion(env, h_core)
            v_obs_smooth = smooth_observed_motion(motion_history[-window:]) 
            
            if any(np.isnan([v_obs_smooth[0], v_obs_smooth[1], v_mean_star[0], v_mean_star[1], v_bunkers[0], v_bunkers[1]])):
                continue

            wind_data_step = props.get('wind_field', props)
            shear_mag = np.sqrt((wind_data_step['u500'] - wind_data_step['u850'])**2 + (wind_data_step['v500'] - wind_data_step['v850'])**2)
            
            if 'w_obs' in o and 'w_mean' in o and 'w_bunkers' in o:
                from StormCast.config import BlendingWeights
                weights = BlendingWeights(w_obs=o['w_obs'], w_mean=o['w_mean'], w_bunkers=o['w_bunkers'])
            else:
                weights = adjust_weights_for_maturity(h_core, len(motion_history), shear_mag)
                
            v_final = blend_motion(v_obs_smooth, v_mean_star, v_bunkers, weights)
            
            process_noise_scale = o.get('process_noise_scale', 1.0)
            kf.predict(dt=dt, process_noise_scale=process_noise_scale)
            kf.update(observation=(cumulative_x, cumulative_y), track_history=len(motion_history))
            
            # Multi-lead time evaluation
            current_state = StormState(
                x=kf.x, y=kf.y, u=v_final[0], v=v_final[1],
                h_core=h_core, track_history=len(motion_history)
            )
            
            lead_times = [900, 1800, 2700, 3600] # 15, 30, 45, 60 min
            forecasts = forecast_with_uncertainty(current_state, lead_times=lead_times)
            target_forecasts = {f.lead_time: f for f in forecasts}

            future_time = 0
            for j in range(i + 1, len(cell_data)):
                future_time += cell_data[j].get('dt', 0)
                
                for lt in lead_times:
                    if abs(future_time - lt) < 150: # +/- 2.5 min window
                        act_x = cumulative_x
                        act_y = cumulative_y
                        for k in range(i+1, j+1):
                            act_x += cell_data[k].get('dx', 0)
                            act_y += cell_data[k].get('dy', 0)
                        
                        f = target_forecasts[lt]
                        dist_err = np.sqrt((act_x - f.x)**2 + (act_y - f.y)**2)
                        
                        key = lt // 60
                        if key not in results:
                            results[key] = []
                            results[f"inside_{key}"] = []
                            results[f"radius_{key}"] = []
                            
                        results[key].append(dist_err)
                        radius = np.sqrt(f.sigma_x * f.sigma_y)
                        results[f"radius_{key}"].append(radius)
                        
                        is_inside = ((act_x - f.x)**2 / f.sigma_x**2) + ((act_y - f.y)**2 / f.sigma_y**2) <= CHI2_95
                        results[f"inside_{key}"].append(bool(is_inside))
                            
                if future_time > max(lead_times) + 300: break

            results['steps_tracked'] += 1
                
        except Exception as e:
            results['errors'].append(f"Error {i}: {str(e)}")
            
    return results

def run_evaluation(data_path, max_files=None, overrides=None):
    """Run evaluation and aggregate metrics."""
    metrics = {
        'total_files': 0,
        'total_steps': 0,
        'error_files': 0,
        'sample_errors': [],
        'intervals': {15: [], 30: [], 45: [], 60: []},
        'inside': {15: [], 30: [], 45: [], 60: []},
        'radius': {15: [], 30: [], 45: [], 60: []}
    }
    
    file_list = glob.glob(os.path.join(data_path, "*", "*.json"))
    if not file_list:
        print(f"No JSON files found in {data_path}")
        return metrics

    if max_files:
        import random
        random.seed(42)
        random.shuffle(file_list)
        file_list = file_list[:max_files]
        
    for file_path in file_list:
        if os.path.basename(file_path) == 'cell_index.json': continue
        res = process_file(file_path, overrides=overrides)
        if res:
            metrics['total_files'] += 1
            metrics['total_steps'] += res['steps_tracked']
            
            for m in [15, 30, 45, 60]:
                if m in res:
                    metrics['intervals'][m].extend(res[m])
                    metrics['inside'][m].extend(res[f"inside_{m}"])
                    metrics['radius'][m].extend(res[f"radius_{m}"])

            if res['errors']:
                metrics['error_files'] += 1
                if len(metrics['sample_errors']) < 10:
                    metrics['sample_errors'].extend([f"{os.path.basename(file_path)}: {e}" for e in res['errors'][:1]])
                    
    return metrics

def main():
    data_path = os.path.expanduser("~/StormCast_Data")
    overrides = {
        'w_obs': 0.4,
        'w_mean': 0.3,
        'w_bunkers': 0.3,
        'window': 9,
        'process_noise_scale': 0.1
    }
    # Process a representative subset of 2000 files for quick result
    metrics = run_evaluation(data_path, max_files=2000, overrides=overrides)
    
    print("\n" + "="*80)
    print(f"Optimized Evaluation Results (2000 files)")
    print(f"Parameters: {overrides}")
    print("-" * 80)
    print(f"{'Lead Time':<10} | {'Hit Rate':<10} | {'Miss Rate':<10} | {'MAE (km)':<10} | {'Avg Radius (km)':<15}")
    print("-" * 80)
    
    for m in [15, 30, 45, 60]:
        errs = metrics['intervals'][m]
        if errs:
            mae = np.mean(errs) / 1000.0
            hit_rate = np.mean(metrics['inside'][m]) * 100.0
            avg_rad = np.mean(metrics['radius'][m]) / 1000.0
            print(f"{m} min     | {hit_rate:.1f}%      | {100-hit_rate:.1f}%      | {mae:.2f}       | {avg_rad:.2f}")
        else:
            print(f"{m} min     | N/A        | N/A        | N/A        | N/A")
            
    print("="*80)
    if metrics['sample_errors']:
        print("\nSample Errors:")
        for err in metrics['sample_errors']:
            print(f"  - {err}")

if __name__ == "__main__":
    main()
