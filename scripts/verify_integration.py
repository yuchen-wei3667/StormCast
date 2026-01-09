#!/usr/bin/env python3
"""
Verification script for StormCastEngine integration.
"""
import sys
import os
from datetime import datetime

# Ensure src is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from StormCast import StormCastEngine, EnvironmentProfile

def verify():
    print("Initializing StormCastEngine...")
    engine = StormCastEngine()
    
    # 1. Set Mock Environment
    print("Setting environment...")
    env = EnvironmentProfile(
        winds={
            850: (10.0, 5.0),
            700: (15.0, 8.0),
            500: (25.0, 10.0),
            250: (40.0, 15.0),
        },
        timestamp=datetime.now()
    )
    engine.set_environment(env)
    
    # 2. Add Mock Observations (moving roughly northeast)
    print("Adding observations...")
    # t=0: (0,0)
    engine.add_observation(x=0.0, y=0.0, dt_seconds=0.0, timestamp=datetime.now())
    
    # t=300s (5 min): (1500, 1500) -> vel ~ (5 m/s, 5 m/s)
    engine.add_observation(x=1500.0, y=1500.0, dt_seconds=300.0, timestamp=datetime.now())
    
    # t=600s (10 min): (3000, 3000) -> vel ~ (5 m/s, 5 m/s)
    engine.add_observation(x=3000.0, y=3000.0, dt_seconds=300.0, timestamp=datetime.now())
    
    # 3. Generate Forecast
    print(f"Kalman P[0][0]: {engine.kalman_filter.P[0][0]:.1f}")
    import math
    print(f"Initial Sigma: {math.sqrt(engine.kalman_filter.P[0][0]):.1f}m")
    
    print("Generating forecast...")
    result = engine.generate_forecast(lead_times=[1800.0, 3600.0])
    
    # 4. Inspect Results
    print("\n--- Verification Results ---")
    print(f"Storm Velocity (Blended): ({result.u:.2f}, {result.v:.2f}) m/s")
    
    print("\nUncertainty Cones:")
    if not result.forecast_cones:
        print("  ERROR: No uncertainty cones generated!")
        sys.exit(1)
        
    for cone in result.forecast_cones:
        lat, lon = cone['center']
        print(f"  T+{cone['lead_time']}s: Center=(Lat {lat:.4f}, Lon {lon:.4f}) Radius={cone['radius']:.1f}m")
        
    print("\nSUCCESS: Integration verified.")

if __name__ == "__main__":
    verify()
