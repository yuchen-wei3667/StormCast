#!/usr/bin/env python3
"""
StormCast Parameter Optimizer

Iterates through different blending weights and uncertainty scaling factors
to find the configuration that minimizes MAE and achieves ~95% coverage on real data.
"""

import sys
import os
import numpy as np

# Add scripts to path to import test_on_real_data
sys.path.insert(0, os.path.dirname(__file__))
from test_on_real_data import run_evaluation

def optimize():
    data_path = "/home/yuchenwei/StormCast/StormCast_Training_Data"
    subset_size = 1000 # Use 1000 files per iteration for speed
    
    print(f"Starting Parameter Optimization (Subset size: {subset_size})...")
    
    # =========================================================================
    # Step 1: Optimize Blending Weights for MAE
    # =========================================================================
    print("\n--- Phase 1: Optimizing Blending Weights for MAE ---")
    
    best_mae = float('inf')
    best_weights = {}
    
    # Try different weight combinations (w_obs, w_mean, w_bunkers)
    # They must sum to 1.0
    weight_samples = [
        (0.3, 0.35, 0.35),
        (0.4, 0.3, 0.3),
        (0.5, 0.25, 0.25),
        (0.6, 0.2, 0.2),
        (0.7, 0.15, 0.15),
        (0.8, 0.1, 0.1),
        (0.9, 0.05, 0.05),
        (0.5, 0.1, 0.4), # Heavier Bunkers
        (0.5, 0.4, 0.1), # Heavier Mean
        (0.2, 0.4, 0.4), # Low Obs
    ]
    
    for wo, wm, wb in weight_samples:
        params = {'w_obs': wo, 'w_mean': wm, 'w_bunkers': wb}
        metrics = run_evaluation(data_path, overrides=params, max_files=subset_size)
        print(f"  Weights (Obs={wo}, Mean={wm}, Bunk={wb}): MAE_next={metrics['mae']:.2f}m, MAE_30m={metrics['mae_30']:.2f}m")
        
        # We optimize weights for a balance, but prioritizing MAE_30m for long-term consistency
        if metrics['mae_30'] < best_mae:
            best_mae = metrics['mae_30']
            best_weights = params
            
    print(f"\nBest Weights found: {best_weights} with MAE: {best_mae:.2f}m")
    
    # =========================================================================
    # Step 2: Optimize Uncertainty Scaling for 95% Coverage
    # =========================================================================
    print("\n--- Phase 2: Optimizing Uncertainty Scaling for Coverage ---")
    
    best_coverage_err = float('inf')
    best_sigma_scale = 1.0
    
    # Try different scaling factors for the uncertainty ellipses
    scales = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    
    for s in scales:
        params = best_weights.copy()
        params['sigma_scale'] = s
        metrics = run_evaluation(data_path, overrides=params, max_files=subset_size)
        print(f"  Sigma Scale {s:.1f}: Coverage={metrics['coverage']:.2f}%")
        
        coverage_err = abs(95.0 - metrics['coverage'])
        if coverage_err < best_coverage_err:
            best_coverage_err = coverage_err
            best_sigma_scale = s
            
    print(f"\nBest Sigma Scale found: {best_sigma_scale:.1f} (Coverage: {95-best_coverage_err:.2f}%)")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "="*40)
    print("Optimization Result:")
    print(f"  w_obs:     {best_weights['w_obs']}")
    print(f"  w_mean:    {best_weights['w_mean']}")
    print(f"  w_bunkers: {best_weights['w_bunkers']}")
    print(f"  sigma_scale: {best_sigma_scale}")
    print("="*40)
    
if __name__ == "__main__":
    optimize()
