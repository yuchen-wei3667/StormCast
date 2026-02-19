#!/usr/bin/env python3
import os
import sys
import numpy as np

# Add scripts to path
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from test_on_real_data import run_evaluation
from StormCast.config import GAUSSIAN_WEIGHT_PARAMS

def optimize_gaussian():
    data_path = os.path.expanduser("~/StormCast_Data")
    
    # Define search space for sigma (spread in km)
    sigma_list = [7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    
    best_mae = float('inf')
    best_sigma = None
    
    results = []
    
    print(f"{'Sigma (km)':<15} | {'MAE (km)':<15} | {'Hit Rate (%)':<15}")
    print("-" * 50)
    
    # Process a representative subset of files for quick result
    for sigma in sigma_list:
        # We need to temporarily modify the GAUSSIAN_WEIGHT_PARAMS in memory
        # to test different sigma values
        original_sigmas = {p: params.sigma for p, params in GAUSSIAN_WEIGHT_PARAMS.items()}
        
        for p in GAUSSIAN_WEIGHT_PARAMS:
            # dataclasses are frozen in config.py, so we have to use object.__setattr__
            object.__setattr__(GAUSSIAN_WEIGHT_PARAMS[p], 'sigma', sigma)
        
        # We use the optimized parameters from the previous run
        overrides = {
            'w_obs': 0.4,
            'w_mean': 0.3,
            'w_bunkers': 0.3,
            'window': 9,
            'process_noise_scale': 0.1
        }
        
        # Run evaluation
        metrics = run_evaluation(data_path, max_files=500, overrides=overrides)
        
        # Restore original sigmas
        for p in GAUSSIAN_WEIGHT_PARAMS:
            object.__setattr__(GAUSSIAN_WEIGHT_PARAMS[p], 'sigma', original_sigmas[p])
            
        # Analyze 30-min MAE
        errs_30 = metrics['intervals'][30]
        if errs_30:
            mae_30 = np.mean(errs_30) / 1000.0
            hit_rate_30 = np.mean(metrics['inside'][30]) * 100.0
            
            print(f"{sigma:<15.1f} | {mae_30:<15.2f} | {hit_rate_30:<15.1f}")
            
            results.append({
                'sigma': sigma,
                'mae_30': mae_30,
                'hit_rate_30': hit_rate_30
            })
            
            if mae_30 < best_mae:
                best_mae = mae_30
                best_sigma = sigma
        else:
            print(f"{sigma:<15.1f} | {'N/A':<15} | {'N/A':<15}")

    print("\n" + "="*40)
    print("Optimization Complete")
    print(f"Best MAE (30 min): {best_mae:.2f} km")
    print(f"Best Sigma: {best_sigma:.1f} km")
    print("="*40)

if __name__ == "__main__":
    optimize_gaussian()
