#!/home/yuchenwei/.conda/envs/StormCast/bin/python3
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from StormCast import StormCastEngine

def verify_filtering():
    engine = StormCastEngine()
    
    # Obs 1: Initial
    engine.add_observation(0, 0, 0)
    
    # Obs 2: Normal motion (10 m/s)
    engine.add_observation(3000, 0, 300) # 10 m/s
    print(f"Motion history after normal obs: {len(engine.motion_history)}")
    
    # Obs 3: Extremely fast (100 m/s)
    engine.add_observation(33000, 0, 300) # (33000-3000)/300 = 100 m/s
    print(f"Motion history after extreme fast: {len(engine.motion_history)}")
    
    # Obs 4: Stationary (0.1 m/s)
    engine.add_observation(33030, 0, 300) # 30/300 = 0.1 m/s
    print(f"Motion history after stationary: {len(engine.motion_history)}")
    
    # Obs 5: Another normal one (5 m/s)
    engine.add_observation(34530, 0, 300) # 1500/300 = 5 m/s
    print(f"Motion history after second normal: {len(engine.motion_history)}")
    
    if len(engine.motion_history) == 2:
        print("\nSUCCESS: Filtering confirmed. Only 2 valid observations recorded.")
    else:
        print(f"\nFAILURE: Filtering failed. History length is {len(engine.motion_history)}")

if __name__ == "__main__":
    verify_filtering()
