#!/usr/bin/env python3
import os
import json
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from StormCast.types import StormState
from StormCast.forecast import forecast_with_uncertainty, serialize_forecast_result

def verify_export():
    # 1. Create a dummy StormState
    state = StormState(
        x=15000.0,
        y=22000.0,
        u=15.5,
        v=8.2,
        h_core=9.5,
        track_history=12,
        motion_jitter=4.5
    )
    
    # 2. Generate lead times
    lead_times = [900, 1800, 2700, 3600]
    
    # 3. Generate forecasts
    forecast_points = forecast_with_uncertainty(state, lead_times=lead_times)
    
    # 4. Serialize to dictionary
    metadata = {
        "id": 12345,
        "timestamp": "2026-01-07T21:40:00Z"
    }
    
    json_entry = serialize_forecast_result(state, forecast_points, cell_metadata=metadata)
    
    # 5. Print to verify
    print("Example JSON Entry (In-Memory Dictionary):")
    print(json.dumps(json_entry, indent=2))

if __name__ == "__main__":
    verify_export()
