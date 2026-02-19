#!/home/yuchenwei/.conda/envs/StormCast/bin/python3
"""
StormCast Cone Size Statistics

Iterates through real data to calculate the typical size (radius) of 
uncertainty cones at different lead times.
"""

import os
import json
import numpy as np
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from StormCast import StormCastEngine, EnvironmentProfile

def get_cone_stats():
    data_path = "/home/yuchenwei/StormCast/StormCast_Training_Data"
    lead_times = [900, 1800, 2700, 3600]
    
    # Store radii for each lead time
    results = {lt: [] for lt in lead_times}
    
    dates = [d for d in sorted(os.listdir(data_path)) if os.path.isdir(os.path.join(data_path, d))]
    
    total_cells = 0
    uncalculable_cells = 0
    total_obs = 0
    filtered_obs = 0
    
    print(f"Analyzing cone sizes across {len(dates)} dates...")
    
    for date in dates:
        date_dir = os.path.join(data_path, date)
        files = [f for f in os.listdir(date_dir) if f.endswith('.json') and f != 'cell_index.json']
        
        for f in files:
            path = os.path.join(date_dir, f)
            with open(path, 'r') as jf:
                cell_data = json.load(jf)
            
            if len(cell_data) < 5:
                continue
            
            total_cells += 1
            
            # Setup Engine
            p0 = cell_data[0]
            engine = StormCastEngine(reference_lat=p0['centroid'][0], reference_lon=p0['centroid'][1])
            
            # Environment
            props0 = p0['properties']
            env = EnvironmentProfile(
                winds={
                    850: (props0['u850'], props0['v850']), 
                    700: (props0['u700'], props0['v700']), 
                    500: (props0['u500'], props0['v500']), 
                    250: (props0['u250'], props0['v250'])
                },
                timestamp=datetime.fromisoformat(p0['timestamp'])
            )
            engine.set_environment(env)
            
            success_count = 0
            # Process history
            for i in range(len(cell_data)):
                step = cell_data[i]
                
                prev_history_len = len(engine.motion_history)
                engine.add_observation(
                    x=step.get('dx', 0), 
                    y=step.get('dy', 0), 
                    dt_seconds=step.get('dt', 0), 
                    echo_top_30=step['properties']['EchoTop30'],
                    echo_top_50=step['properties']['EchoTop50'],
                    timestamp=datetime.fromisoformat(step['timestamp'])
                )
                
                if i > 0:
                    total_obs += 1
                    if len(engine.motion_history) == prev_history_len:
                        filtered_obs += 1
                
                # Try to generate forecast
                if len(engine.motion_history) >= 1:
                    try:
                        res = engine.generate_forecast(lead_times=lead_times, confidence=0.90)
                        for cone in res.forecast_cones:
                            results[cone['lead_time']].append(cone['radius'])
                        success_count += 1
                    except:
                        pass
            
            if success_count == 0:
                uncalculable_cells += 1

    print(f"\nEvaluation Results:")
    print(f"  Total Cells Processed:   {total_cells}")
    print(f"  Uncalculable Cells:      {uncalculable_cells} ({uncalculable_cells/total_cells*100:.1f}%)")
    print(f"  Total Obs Potential:     {total_obs}")
    print(f"  Filtered (Hard Jump):    {filtered_obs} ({filtered_obs/total_obs*100:.1f}%)")
    print("-" * 65)
    print(f"{'Lead Time':<12} | {'Mean (km)':<10} | {'Median (km)':<11} | {'Min (km)':<10} | {'Max (km)':<10}")
    print("-" * 65)
    
    for lt in lead_times:
        radii = np.array(results[lt]) / 1000.0 # Convert to km
        if len(radii) > 0:
            print(f"{int(lt/60):>2} min      | {np.mean(radii):<10.2f} | {np.median(radii):<11.2f} | {np.min(radii):<10.2f} | {np.max(radii):<10.2f}")
        else:
            print(f"{int(lt/60):>2} min      | N/A")

if __name__ == "__main__":
    get_cone_stats()
