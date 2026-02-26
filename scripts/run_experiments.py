import sys
import os

# Add src and scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import StormCast.forecast
from evaluate_forecast_accuracy import evaluate_coverage
import numpy as np

# Original forecast_position
original_forecast_position = StormCast.forecast.forecast_position

def run_experiment(use_accel: bool, use_offset: bool):
    # Monkey-patch forecast_position
    def patched_forecast_position(state, dt):
        ou = state.ou if use_offset else 0.0
        ov = state.ov if use_offset else 0.0
        ax = state.ax if use_accel else 0.0
        ay = state.ay if use_accel else 0.0
        
        if getattr(patched_forecast_position, 'call_count', 0) < 5 and use_accel:
            print(f"DEBUG {use_accel}/{use_offset}: state.ax={state.ax:.4f}, state.ay={state.ay:.4f}, state.ou={state.ou:.4f}, state.ov={state.ov:.4f}")
            patched_forecast_position.call_count = getattr(patched_forecast_position, 'call_count', 0) + 1
            
        x_forecast = state.x + (state.u + ou) * dt + 0.5 * ax * dt * dt
        y_forecast = state.y + (state.v + ov) * dt + 0.5 * ay * dt * dt
        return (x_forecast, y_forecast)
        
    patched_forecast_position.call_count = 0
    StormCast.forecast.forecast_position = patched_forecast_position
    
    print(f"\n--- Running Experiment: use_accel={use_accel}, use_offset={use_offset} ---")
    results = evaluate_coverage("/home/yuchenwei/StormCast_Data", max_files=1000)
    
    print(f"{'Lead':<8} | {'Hit Rate':<10} | {'Miss Rate':<10} | {'MAE (km)':<10} | {'Avg Cone (km)':<12}")
    print("-" * 70)
    
    total_mae = 0
    count = 0
    
    for lt in sorted(results.keys()):
        h, t = results[lt]['hits'], results[lt]['total']
        if t == 0: continue
        
        hit_rate = h / t
        miss_rate = 1.0 - hit_rate
        mae = np.mean(results[lt]['mae'])
        avg_sigma = np.mean(results[lt]['sigma_avg'])
        avg_radius = avg_sigma * np.sqrt(5.991)
        
        total_mae += mae
        count += 1
        
        print(f"{lt//60:>2} min   | {hit_rate:>8.1%} | {miss_rate:>8.1%} | {mae:>8.2f} | {avg_radius:>8.2f}")
    
    if count > 0:
        print(f"Average MAE across all lead times: {total_mae/count:.2f} km")

if __name__ == "__main__":
    run_experiment(False, False)
    run_experiment(True, False)
    run_experiment(False, True)
    run_experiment(True, True)
