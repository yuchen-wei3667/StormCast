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
    Process a single storm cell JSON file with optional parameter overrides.
    
    overrides: dict containing keys like 'w_obs', 'sigma_vel', etc.
    """
    with open(file_path, 'r') as f:
        cell_data = json.load(f)
    
    if not isinstance(cell_data, list) or len(cell_data) < 2:
        return None

    results = {
        'errors': [],
        'forecast_errors': [],
        'inside_range': [],
        'steps_tracked': 0
    }
    
    # Apply overrides to local variables or config-derived values
    o = overrides or {}
    
    # Initialize
    first_step = cell_data[0]
    props = first_step['properties']
    
    # Prepare environment
    env = EnvironmentProfile(
        winds={
            850: (props['u850'], props['v850']),
            700: (props['u700'], props['v700']),
            500: (props['u500'], props['v500']),
            250: (props['u250'], props['v250']),
        },
        timestamp=datetime.fromisoformat(first_step['timestamp'])
    )
    
    # Kalman initialization with overrides
    # If we want to override sigma_pos, sigma_vel, etc, we'd need to pass them to KF
    # For now let's focus on the blending weights and basic uncertainty scaling
    kf = StormKalmanFilter(initial_state=[0.0, 0.0, 0.0, 0.0])
    
    motion_history = []
    cumulative_x = 0.0
    cumulative_y = 0.0
    
    last_forecast = None
    CHI2_95 = 5.991
    
    for i, step in enumerate(cell_data):
        props = step['properties']
        dx = step.get('dx')
        dy = step.get('dy')
        dt = step.get('dt')
        
        if dx is not None: cumulative_x += dx
        if dy is not None: cumulative_y += dy
        
        if last_forecast is not None:
            fx, fy, fsx, fsy = last_forecast
            # Apply uncertainty scaling if provided
            fsx_scaled = fsx * o.get('sigma_scale', 1.0)
            fsy_scaled = fsy * o.get('sigma_scale', 1.0)
            
            dist_err = np.sqrt((cumulative_x - fx)**2 + (cumulative_y - fy)**2)
            results['forecast_errors'].append(dist_err)
            
            if fsx_scaled > 0 and fsy_scaled > 0:
                is_inside = ((cumulative_x - fx)**2 / fsx_scaled**2) + ((cumulative_y - fy)**2 / fsy_scaled**2) <= CHI2_95
                results['inside_range'].append(is_inside)
            else:
                results['inside_range'].append(dist_err < 1000)

        if dx is None or dy is None or dt is None or dt == 0:
            last_forecast = None
            continue
            
        u_obs_raw = dx / dt
        v_obs_raw = dy / dt
        motion_history.append((u_obs_raw, v_obs_raw))
        
        try:
            h_core = compute_storm_core_height(props['EchoTop30'], props['EchoTop50'])
            v_mean_star = compute_adaptive_steering(env, h_core)
            v_bunkers = compute_bunkers_motion(env, h_core)
            v_obs_smooth = smooth_observed_motion(motion_history[-5:]) 
            
            if any(np.isnan([v_obs_smooth[0], v_obs_smooth[1], v_mean_star[0], v_mean_star[1], v_bunkers[0], v_bunkers[1]])):
                results['errors'].append(f"NaN at step {i}")
                last_forecast = None
                continue

            shear_mag = np.sqrt((props['u500'] - props['u850'])**2 + (props['v500'] - props['v850'])**2)
            
            # Blending weights with overrides
            if 'w_obs' in o and 'w_mean' in o and 'w_bunkers' in o:
                from StormCast.config import BlendingWeights
                weights = BlendingWeights(w_obs=o['w_obs'], w_mean=o['w_mean'], w_bunkers=o['w_bunkers'])
            else:
                weights = adjust_weights_for_maturity(h_core, len(motion_history), shear_mag)
            
            v_final = blend_motion(v_obs_smooth, v_mean_star, v_bunkers, weights)
            
            kf.predict(dt=dt)
            kf.update(observation=(cumulative_x, cumulative_y), track_history=len(motion_history))
            
            if any(np.isnan(kf.state)):
                results['errors'].append(f"KF NaN at step {i}")
                last_forecast = None
                break
            
            # 4. Generate forecast for the NEXT scan (MAE evaluation)
            next_step_idx = i + 1
            if next_step_idx < len(cell_data):
                next_step = cell_data[next_step_idx]
                next_dt = next_step.get('dt', dt)
                current_state = StormState(
                    x=kf.x, y=kf.y, u=v_final[0], v=v_final[1],
                    h_core=h_core, track_history=len(motion_history)
                )
                forecasts = forecast_with_uncertainty(current_state, lead_times=[next_dt])
                f = forecasts[0]
                last_forecast = (f.x, f.y, f.sigma_x, f.sigma_y)
            else:
                last_forecast = None
            
            # 5. NEW: Long-Term Forecast Evaluation (30 min)
            # Find a step roughly 30 minutes (1800s) ahead
            target_lead = 1800.0
            future_time = 0
            for j in range(i + 1, len(cell_data)):
                future_time += cell_data[j].get('dt', 0)
                if abs(future_time - target_lead) < 150: # Close enough to 30 min
                    target_step = cell_data[j]
                    # Calculate actual displacement at that time
                    act_x = cumulative_x
                    act_y = cumulative_y
                    for k in range(i+1, j+1):
                        act_x += cell_data[k].get('dx', 0)
                        act_y += cell_data[k].get('dy', 0)
                    
                    # Generate 30m forecast
                    f30 = forecast_with_uncertainty(current_state, lead_times=[future_time])[0]
                    dist_30 = np.sqrt((act_x - f30.x)**2 + (act_y - f30.y)**2)
                    
                    if 'mae_30' not in results: results['mae_30'] = []
                    results['mae_30'].append(dist_30)
                    break
                if future_time > target_lead + 300: break
                
            results['steps_tracked'] += 1
                
        except Exception as e:
            results['errors'].append(f"Error {i}: {str(e)}")
            last_forecast = None
            
    return results

def run_evaluation(data_dir, overrides=None, max_files=None):
    """Run full evaluation and return metrics."""
    metrics = {
        'total_files': 0,
        'total_steps': 0,
        'total_dist_err': 0.0,
        'total_inside': 0,
        'total_forecasts': 0,
        'error_files': 0,
        'total_dist_err_30': 0.0,
        'total_forecasts_30': 0
    }
    
    file_list = []
    for date_folder in sorted(os.listdir(data_dir)):
        date_path = os.path.join(data_dir, date_folder)
        if not os.path.isdir(date_path): continue
        for json_file in os.listdir(date_path):
            if json_file.endswith('.json') and json_file != 'cell_index.json':
                file_list.append(os.path.join(date_path, json_file))
    
    if max_files:
        import random
        random.seed(42)
        random.shuffle(file_list)
        file_list = file_list[:max_files]
        
    for file_path in file_list:
        res = process_file(file_path, overrides=overrides)
        if res:
            metrics['total_files'] += 1
            metrics['total_steps'] += res['steps_tracked']
            metrics['total_dist_err'] += sum(res['forecast_errors'])
            metrics['total_inside'] += sum(res['inside_range'])
            metrics['total_forecasts'] += len(res['forecast_errors'])
            
            if 'mae_30' in res:
                metrics['total_dist_err_30'] += sum(res['mae_30'])
                metrics['total_forecasts_30'] += len(res['mae_30'])

            if res['errors']:
                metrics['error_files'] += 1
                
    if metrics['total_forecasts'] > 0:
        metrics['mae'] = metrics['total_dist_err'] / metrics['total_forecasts']
        metrics['coverage'] = (metrics['total_inside'] / metrics['total_forecasts']) * 100
    else:
        metrics['mae'] = 0
        metrics['coverage'] = 0

    if metrics['total_forecasts_30'] > 0:
        metrics['mae_30'] = metrics['total_dist_err_30'] / metrics['total_forecasts_30']
    else:
        metrics['mae_30'] = 0
        
    return metrics

def main():
    data_path = "/home/yuchenwei/StormCast/StormCast_Training_Data"
    # Process a representative subset of 2000 files for quick result
    metrics = run_evaluation(data_path, max_files=2000)
    
    print("\n" + "="*40)
    print(f"Subset Evaluation Results (2000 files):")
    print(f"MAE: {metrics['mae']:.2f} meters")
    print(f"Coverage: {metrics['coverage']:.2f}%")
    print(f"Total Forecasts: {metrics['total_forecasts']}")
    print("="*40)

if __name__ == "__main__":
    main()

