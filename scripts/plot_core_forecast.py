#!/home/yuchenwei/.conda/envs/StormCast/bin/python
"""
Plot StormCast Forecast

Generates a synthetic storm cell, processes it with StormCastEngine,
and plots the resulting forecast track and uncertainty cones.
"""

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
    # 1. Setup Data Paths
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "storm_forecast.png")

    # 2. Initialize Engine
    ref_lat, ref_lon = 35.0, -97.0
    engine = StormCastEngine(reference_lat=ref_lat, reference_lon=ref_lon)

    # 3. Define Environment (Meteorologically consistent)
    # Strong southwesterly flow
    env = EnvironmentProfile(
        winds={
            850: (12.0, 10.0),
            700: (18.0, 12.0),
            500: (28.0, 15.0),
            250: (45.0, 15.0),
        },
        timestamp=datetime.now()
    )
    engine.set_environment(env)

    # 4. Generate Synthetic Observations (moving ~NE)
    u_true, v_true = 15.0, 6.0 # m/s
    dt = 300 # 5 minutes
    
    past_lats = []
    past_lons = []
    
    print("Feeding observations to engine...")
    for i in range(10):
        # Calculate displacement in meters
        x_m = u_true * i * dt
        y_m = v_true * i * dt
        
        # Add a bit of noise to position
        import random
        x_m += random.uniform(-100, 100)
        y_m += random.uniform(-100, 100)
        
        engine.add_observation(
            x=x_m, 
            y=y_m, 
            dt_seconds=dt if i > 0 else 0,
            echo_top_30=12.0,
            echo_top_50=9.0,
            timestamp=datetime.now()
        )
        
        # Track for plotting
        lat, lon = engine._meters_to_latlon(x_m, y_m)
        past_lats.append(lat)
        past_lons.append(lon)

    # 5. Generate Forecast
    print("Generating forecast...")
    lead_times = [900, 1800, 2700, 3600] # 15, 30, 45, 60 min
    result = engine.generate_forecast(lead_times=lead_times, confidence=0.95)

    # 6. Plotting
    print(f"Plotting to {output_file}...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    
    # Aesthetics
    ax.set_facecolor('#0b0e14')
    fig.patch.set_facecolor('#0b0e14')
    
    # Plot Past Track
    ax.plot(past_lons, past_lats, color='#00d1ff', marker='o', markersize=4, 
            linewidth=2, alpha=0.8, label='Observed Track')
    ax.scatter(past_lons[-1], past_lats[-1], color='#ffff00', s=100, 
               edgecolor='white', zorder=5, label='Current Position')
    
    # Plot Forecast
    colors = ['#ffe100', '#ffae00', '#ff7b00', '#ff3c00']
    
    for i, cone in enumerate(result.forecast_cones):
        c_lat, c_lon = cone['center']
        radius_m = cone['radius']
        
        # Convert radius from meters to degrees (approximate)
        radius_deg = radius_m / 111111.0
        
        # Plot Confidence Circle
        circle = Circle((c_lon, c_lat), radius_deg, color=colors[i], 
                        alpha=0.15, zorder=2)
        ax.add_patch(circle)
        ax.plot([], []) # Dummy for legend if needed
        
        # Plot circle outline
        circle_outline = Circle((c_lon, c_lat), radius_deg, color=colors[i], 
                               fill=False, linestyle='--', alpha=0.4, linewidth=1, zorder=3)
        ax.add_patch(circle_outline)
        
        # Plot center
        ax.scatter(c_lon, c_lat, color=colors[i], marker='x', s=40, 
                   zorder=4, label=f'T+{cone["lead_time"]//60}m' if i == 0 or i == 3 else "")

    # Labels and Grid
    ax.set_title("StormCast Integrated Forecast - Synthetic Supercell", 
                 color='white', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Longitude [deg]", color='#888888')
    ax.set_ylabel("Latitude [deg]", color='#888888')
    
    ax.grid(True, linestyle=':', alpha=0.2, color='white')
    ax.set_aspect(1.0 / math.cos(math.radians(ref_lat))) # Correct aspect ratio
    
    # Legend
    legend = ax.legend(frameon=True, facecolor='#161b22', edgecolor='#30363d', loc='upper left')
    for text in legend.get_texts():
        text.set_color('white')
        
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor=fig.get_facecolor())
    print("Done.")

if __name__ == "__main__":
    main()
