#!/usr/bin/env python3
import os
import sys
import numpy as np
from itertools import product

# Add scripts to path
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from test_on_real_data import run_evaluation

def optimize():
    data_path = os.path.expanduser("~/StormCast_Data")
    
    # Define comprehensive 3D search space
    w_obs_list = [0.2, 0.4, 0.6, 0.8]
    window_list = [5, 7, 9, 11]
    pnoise_list = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    best_mae = float('inf')
    best_params = {}
    
    results = []
    
    print(f"Starting Grid Search with {len(w_obs_list)*len(window_list)*len(pnoise_list)} combinations...")
    print(f"{'w_obs':<10} | {'window':<10} | {'pnoise':<10} | {'MAE (km)':<15} | {'Hit Rate (%)':<15}")
    print("-" * 65)
    
    for w_obs, window, pnoise in product(w_obs_list, window_list, pnoise_list):
        # Keep mean and bunkers equal
        w_rest = (1.0 - w_obs) / 2.0
        
        overrides = {
            'w_obs': w_obs,
            'w_mean': w_rest,
            'w_bunkers': w_rest,
            'window': window,
            'process_noise_scale': pnoise
        }
        
        # Run on a smaller subset (500 files) for speed during grid search
        metrics = run_evaluation(data_path, max_files=500, overrides=overrides)
        
        # We optimize based on 30-min MAE
        errs_30 = metrics['intervals'][30]
        if errs_30:
            mae_30 = np.mean(errs_30) / 1000.0
            hit_rate_30 = np.mean(metrics['inside'][30]) * 100.0
            
            print(f"{w_obs:<10.1f} | {window:<10} | {pnoise:<10.2f} | {mae_30:<15.2f} | {hit_rate_30:<15.1f}")
            
            results.append({
                'w_obs': w_obs,
                'window': window,
                'pnoise': pnoise,
                'mae_30': mae_30,
                'hit_rate_30': hit_rate_30
            })
            
            if mae_30 < best_mae:
                best_mae = mae_30
                best_params = overrides
        else:
            print(f"{w_obs:<10.1f} | {window:<10} | {pnoise:<10.2f} | {'N/A':<15} | {'N/A':<15}")

    print("\n" + "="*40)
    print("Optimization Complete")
    print(f"Best MAE (30 min): {best_mae:.2f} km")
    print(f"Best Parameters: {best_params}")
    print("="*40)

if __name__ == "__main__":
    optimize()
