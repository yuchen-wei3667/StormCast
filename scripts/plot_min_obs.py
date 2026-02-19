#!/home/yuchenwei/.conda/envs/StormCast/bin/python3
import os
import sys
import math
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from StormCast import StormCastEngine, EnvironmentProfile

def main():
    # 1. Setup
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "storm_forecast.png")

    # 2. Initialize Engine
    ref_lat, ref_lon = 35.0, -97.0
    engine = StormCastEngine(reference_lat=ref_lat, reference_lon=ref_lon)

    # 3. Environment
    env = EnvironmentProfile(
        winds={850: (10, 5), 700: (15, 7), 500: (20, 10), 250: (30, 15)},
        timestamp=datetime.now()
    )
    engine.set_environment(env)

    # 4. Minimal Observations
    # Position 1: Origin
    print("Scan 1: Adding observation 1 at (0,0)")
    engine.add_observation(x=0, y=0, dt_seconds=0)
    
    # Position 2: 3km North after 5 mins (10 m/s)
    print("Scan 2: Adding observation 2 at (0, 3000) after 300s")
    engine.add_observation(x=0, y=3000, dt_seconds=300)

    # 5. Generate Forecast
    print(f"Motion history length: {len(engine.motion_history)}")
    lead_times = [900, 1800, 2700, 3600]
    result = engine.generate_forecast(lead_times=lead_times, confidence=0.90)

    # 6. Plotting
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    ax.set_facecolor('#0b0e14')
    fig.patch.set_facecolor('#0b0e14')
    
    # Past (only 2 points)
    past_x = [0, 0]
    past_y = [0, 3000]
    past_lats, past_lons = [], []
    for x, y in zip(past_x, past_y):
        lat, lon = engine._meters_to_latlon(x, y)
        past_lats.append(lat)
        past_lons.append(lon)
        
    ax.plot(past_lons, past_lats, color='#00d1ff', marker='o', label='Observed (2 points)')
    ax.scatter(past_lons[-1], past_lats[-1], color='#ffff00', s=100, zorder=5)
    
    # Cones
    colors = ['#ffe100', '#ffae00', '#ff7b00', '#ff3c00']
    for i, cone in enumerate(result.forecast_cones):
        c_lat, c_lon = cone['center']
        radius_deg = cone['radius'] / 111111.0
        circle = Circle((c_lon, c_lat), radius_deg, color=colors[i], alpha=0.15, zorder=2)
        ax.add_patch(circle)
        ax.scatter(c_lon, c_lat, color=colors[i], marker='x', s=40)

    ax.set_title("StormCast Minimal Forecast (N=2 Scans)", color='white', fontsize=16)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect(1.0 / math.cos(math.radians(ref_lat)))
    ax.grid(True, alpha=0.1)
    
    plt.savefig(output_file, facecolor=fig.get_facecolor())
    print(f"Success. Plot saved to {output_file}")

if __name__ == "__main__":
    main()
