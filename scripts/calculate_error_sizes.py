#!/usr/bin/env python3
import math
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from StormCast.config import UNCERTAINTY_PARAMS, DEFAULT_LEAD_TIMES
from StormCast.uncertainty import compute_tracking_uncertainty, compute_velocity_covariance

def calculate_sizes(n_samples=10, jitter=2.0):
    sigma_u, sigma_v, sigma_total = compute_velocity_covariance(
        n_samples=n_samples,
        motion_jitter=jitter
    )
    
    # 95% confidence radius factor for 2D isotropic Gaussian (sqrt(chi2.ppf(0.95, df=2)))
    # chi2.ppf(0.95, 2) is approx 5.991
    radius_factor = math.sqrt(5.991) # approx 2.447
    
    print(f"Parameters: sigma_obs={UNCERTAINTY_PARAMS.sigma_obs}, sigma_env={UNCERTAINTY_PARAMS.sigma_env}, jitter={jitter}")
    print(f"Derived Sigma_vel: {sigma_total:.2f} m/s")
    print("-" * 40)
    print(f"{'Lead (min)':<12} | {'Sigma_pos (km)':<15} | {'95% Radius (km)':<15}")
    print("-" * 40)
    
    for dt in DEFAULT_LEAD_TIMES:
        # sigma_pos(T) = sigma_total * T
        sigma_pos = sigma_total * dt / 1000.0
        radius = sigma_pos * radius_factor
        print(f"{int(dt/60):<12} | {sigma_pos:<15.2f} | {radius:<15.2f}")

if __name__ == "__main__":
    calculate_sizes()
