#!/usr/bin/env python3
"""
StormCast Demo Script

Demonstrates the complete StormCast pipeline with synthetic data:
1. Data Acquisition (mock storm + environment)
2. Height-Dependent Weights
3. Environmental Diagnostics
4. Motion Smoothing & Blending
5. Kalman Filter State Estimation
6. Forecast Generation with Uncertainty
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime import datetime, timedelta

from StormCast.types import StormState, EnvironmentProfile
from StormCast.config import (
    PRESSURE_LEVELS,
    DEFAULT_BLENDING_WEIGHTS,
    GAUSSIAN_WEIGHT_PARAMS,
    BUNKERS_DEVIATION,
)
from StormCast.diagnostics import (
    compute_height_weights,
    compute_adaptive_steering,
    compute_effective_shear,
    compute_bunkers_motion,
    compute_storm_core_height,
)
from StormCast.blending import (
    smooth_observed_motion,
    compute_propagation,
    blend_motion,
    adjust_weights_for_maturity,
)
from StormCast.kalman import StormKalmanFilter
from StormCast.uncertainty import (
    compute_tracking_uncertainty,
    compute_velocity_covariance,
)
from StormCast.forecast import (
    forecast_position,
    generate_forecast_track,
    forecast_with_uncertainty,
)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def print_vector(name: str, u: float, v: float, unit: str = "m/s"):
    """Print a formatted vector."""
    import math
    mag = math.sqrt(u*u + v*v)
    direction = (270.0 - math.degrees(math.atan2(v, u))) % 360.0
    print(f"  {name}: ({u:.2f}, {v:.2f}) {unit}")
    print(f"    Magnitude: {mag:.2f} {unit}, Direction: {direction:.1f}°")


def main():
    print_section("StormCast Demo - Synthetic Storm Scenario")
    
    # =========================================================================
    # Step 1: Create Synthetic Data
    # =========================================================================
    print_section("Step 1: Data Acquisition")
    
    # Synthetic RAP wind profile (typical spring storm environment)
    # Southwesterly flow increasing with height
    environment = EnvironmentProfile(
        winds={
            850: (10.0, 8.0),   # ~13 m/s from SSW at low levels
            700: (15.0, 10.0), # ~18 m/s from SW
            500: (25.0, 12.0), # ~28 m/s from WSW at steering level
            250: (40.0, 5.0),  # ~40 m/s from W at jet level
        },
        timestamp=datetime.now()
    )
    
    print("  RAP Wind Profile:")
    for level in sorted(environment.winds.keys(), reverse=True):
        u, v = environment.winds[level]
        print(f"    {level} mb: u={u:.1f} m/s, v={v:.1f} m/s")
    
    # Synthetic storm echo tops
    echo_top_30 = 12.0  # km AGL - 30 dBZ top
    echo_top_50 = 10.0  # km AGL - 50 dBZ top
    h_core = compute_storm_core_height(echo_top_30, echo_top_50)
    print(f"\n  Echo Tops: 30dBZ={echo_top_30} km, 50dBZ={echo_top_50} km")
    print(f"  Storm Core Height: {h_core:.1f} km AGL (moderate-deep convection)")
    
    # Observed storm motion history (simulating radar tracking)
    # 5 observations showing rightward deviant motion
    motion_history = [
        (22.0, 8.0),   # oldest
        (23.0, 7.5),
        (24.0, 7.0),
        (23.5, 6.5),
        (24.5, 6.0),   # most recent
    ]
    print("\n  Observed Motion History (u, v in m/s):")
    for i, (u, v) in enumerate(motion_history, 1):
        print(f"    t-{len(motion_history)-i}: ({u:.1f}, {v:.1f})")
    
    # =========================================================================
    # Step 2: Height-Dependent Weights
    # =========================================================================
    print_section("Step 2: Height-Dependent Weight Calculation")
    
    weights = compute_height_weights(h_core)
    print(f"  Gaussian weights for H_core = {h_core:.1f} km:")
    for level in PRESSURE_LEVELS:
        params = GAUSSIAN_WEIGHT_PARAMS[level]
        print(f"    {level} mb: w={weights[level]:.3f} (μ={params.mu} km, σ={params.sigma} km)")
    
    # =========================================================================
    # Step 3: Environmental Diagnostics
    # =========================================================================
    print_section("Step 3: Environmental Diagnostics")
    
    # Height-adaptive mean wind
    v_mean_star = compute_adaptive_steering(environment, h_core)
    print_vector("Height-Adaptive Mean Wind (V_mean*)", v_mean_star[0], v_mean_star[1])
    
    # Effective shear
    shear = compute_effective_shear(environment, h_core)
    print_vector("Effective Shear (S_eff)", shear[0], shear[1])
    
    # Bunkers right-mover motion
    v_bunkers = compute_bunkers_motion(environment, h_core, right_mover=True)
    print_vector("Bunkers Right-Mover (V_bunkers*)", v_bunkers[0], v_bunkers[1])
    
    d_mag = BUNKERS_DEVIATION.d_shallow + (BUNKERS_DEVIATION.d_deep - BUNKERS_DEVIATION.d_shallow) * \
            min(1.0, max(0.0, (h_core - BUNKERS_DEVIATION.h_shallow) / 
                         (BUNKERS_DEVIATION.h_deep - BUNKERS_DEVIATION.h_shallow)))
    print(f"    Deviation Magnitude D = {d_mag:.2f} m/s")
    
    # =========================================================================
    # Step 4: Motion Smoothing & Blending
    # =========================================================================
    print_section("Step 4: Motion Smoothing & Blending")
    
    # Smooth observed motion
    v_obs = smooth_observed_motion(motion_history, method="exponential", alpha=0.3)
    print_vector("Smoothed Observed Motion (V_obs)", v_obs[0], v_obs[1])
    
    # Propagation component
    v_prop = compute_propagation(v_obs, v_mean_star)
    print_vector("Propagation (V_prop = V_obs - V_mean*)", v_prop[0], v_prop[1])
    
    # Dynamic weight adjustment
    import math
    shear_mag = math.sqrt(shear[0]**2 + shear[1]**2)
    adjusted_weights = adjust_weights_for_maturity(
        h_core=h_core,
        track_history=len(motion_history),
        shear_magnitude=shear_mag,
    )
    print(f"\n  Adjusted Blending Weights (track_history={len(motion_history)}, shear={shear_mag:.1f} m/s):")
    print(f"    w_obs={adjusted_weights.w_obs:.3f}, w_mean={adjusted_weights.w_mean:.3f}, w_bunkers={adjusted_weights.w_bunkers:.3f}")
    
    # Final blended motion
    v_final = blend_motion(v_obs, v_mean_star, v_bunkers, weights=adjusted_weights)
    print_vector("\nBlended Motion (V_final)", v_final[0], v_final[1])
    
    # =========================================================================
    # Step 5: Kalman Filter State Estimation
    # =========================================================================
    print_section("Step 5: Kalman Filter State Estimation")
    
    # Initialize filter with current position (hypothetical)
    initial_state = [0.0, 0.0, v_final[0], v_final[1]]  # position at origin, velocity from blending
    kf = StormKalmanFilter(initial_state=initial_state)
    
    print("  Initial State:")
    print(f"    Position: ({kf.x:.0f} m, {kf.y:.0f} m)")
    print(f"    Velocity: ({kf.u:.2f} m/s, {kf.v:.2f} m/s)")
    
    # Simulate a few predict-update cycles
    dt = 300  # 5 minutes
    simulated_observations = [
        (7200.0, 1800.0),   # Position after 5 min
        (14500.0, 3500.0),  # Position after 10 min
        (21900.0, 5100.0),  # Position after 15 min
    ]
    
    print(f"\n  Simulating {len(simulated_observations)} observation cycles (dt={dt}s)...")
    for i, obs in enumerate(simulated_observations, 1):
        kf.predict(dt=dt)
        kf.update(observation=obs, track_history=len(motion_history)+i)
        print(f"    After update {i}: pos=({kf.x:.0f}, {kf.y:.0f}) m, vel=({kf.u:.2f}, {kf.v:.2f}) m/s")
    
    # =========================================================================
    # Step 6: Forecast Generation
    # =========================================================================
    print_section("Step 6: Forecast Generation with Uncertainty")
    
    # Create storm state from Kalman filter output
    storm = StormState(
        x=kf.x,
        y=kf.y,
        u=kf.u,
        v=kf.v,
        h_core=h_core,
        track_history=len(motion_history) + len(simulated_observations),
        timestamp=datetime.now(),
    )
    
    print(f"  Current Storm State:")
    print(f"    Position: ({storm.x:.0f} m, {storm.y:.0f} m)")
    print(f"    Velocity: ({storm.u:.2f} m/s, {storm.v:.2f} m/s) = {storm.speed:.2f} m/s @ {storm.direction:.1f}°")
    print(f"    Track History: {storm.track_history} samples")
    
    # Generate forecast with uncertainty
    forecast = forecast_with_uncertainty(storm)
    
    print("\n  Forecast Positions (with uncertainty):")
    print(f"  {'Lead Time':<12} {'X (km)':<12} {'Y (km)':<12} {'σ_x (km)':<12} {'σ_y (km)':<12}")
    print("  " + "-"*52)
    for fp in forecast:
        lead_min = fp.lead_time / 60
        print(f"  {lead_min:>6.0f} min   {fp.x/1000:>8.2f}     {fp.y/1000:>8.2f}     "
              f"{fp.sigma_x/1000:>8.2f}     {fp.sigma_y/1000:>8.2f}")
    
    # Uncertainty info
    sigma_u, sigma_v, _ = compute_velocity_covariance(n_samples=storm.track_history)
    tracking_unc = compute_tracking_uncertainty(storm.track_history)
    print(f"\n  Uncertainty Components:")
    print(f"    Tracking History Uncertainty: {tracking_unc:.2f} m/s (N={storm.track_history})")
    print(f"    Combined Velocity Uncertainty: σ_u = σ_v = {sigma_u:.2f} m/s")
    
    print_section("Demo Complete")
    print("  This demo showed the complete StormCast pipeline:")
    print("    1. Data acquisition from RAP winds and radar echo tops")
    print("    2. Gaussian height-weight calculation")
    print("    3. Height-adaptive steering, shear, and Bunkers motion")
    print("    4. Motion smoothing and weighted blending")
    print("    5. Kalman filter state estimation with forecast smoothing")
    print("    6. Forecast generation with uncertainty propagation")


if __name__ == "__main__":
    main()
