#!/usr/bin/env python3
import os
import json
import argparse
import glob
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not found. Please install it to use the plotting functionality.")
    plt = None

def load_data(data_dir):
    """
    Loads storm cell data from a directory structure:
    data_dir/YYYYMMDD/*.json
    Excludes cell_index.json.
    Calculates u and v from dx, dy, and dt.
    """
    u_vals = []
    v_vals = []
    
    # Traverse YYYYMMDD folders
    date_folders = sorted(glob.glob(os.path.join(data_dir, "[0-9]"*8)))
    if not date_folders:
        print(f"No YYYYMMDD folders found in {data_dir}")
        return np.array([]), np.array([])

    for date_folder in date_folders:
        if not os.path.isdir(date_folder):
            continue
            
        json_files = glob.glob(os.path.join(date_folder, "*.json"))
        for json_file in json_files:
            if os.path.basename(json_file) == "cell_index.json":
                continue
                
            try:
                with open(json_file, 'r') as f:
                    cells = json.load(f)
                
                # Ensure cells is a list
                if not isinstance(cells, list):
                    cells = [cells]
                    
                for cell in cells:
                    dx = cell.get('dx')
                    dy = cell.get('dy')
                    dt = cell.get('dt')
                    
                    if dx is not None and dy is not None and dt is not None and dt != 0:
                        u = dx / dt
                        v = dy / dt
                        
                        u_vals.append(u)
                        v_vals.append(v)
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
                
    return np.array(u_vals), np.array(v_vals)

def plot_uv_scatter(u, v, output_path=None):
    """
    Plots the u and v components on a scatter plot.
    """
    if plt is None:
        print("Error: matplotlib is required for plotting.")
        return

    plt.figure(figsize=(10, 10))
    
    # Create scatter plot of U vs V
    plt.scatter(u, v, color='blue', alpha=0.5, edgecolor='none', s=20)
    
    # Add origin lines
    plt.axhline(0, color='black', linewidth=1, alpha=0.3)
    plt.axvline(0, color='black', linewidth=1, alpha=0.3)
    
    # Set equal scaling to see angles correctly
    plt.axis('equal')
    
    # Add title and labels
    plt.title('Storm Motion (U vs V Vectors)')
    plt.xlabel('U (West-East) Motion [m/s]')
    plt.ylabel('V (South-North) Motion [m/s]')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot scatter of storm U and V motion vectors.")
    parser.add_argument("--data_dir", help="Base directory containing YYYYMMDD subfolders.")
    parser.add_argument("--output", help="Optional path to save the generated plot.")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
    else:
        print(f"Loading data from {args.data_dir}...")
        u, v = load_data(args.data_dir)
        
        if len(u) > 0:
            print(f"Found {len(u)} storm cells. Generating scatter plot...")
            
            # Calculate speed and print statistics
            speed = np.sqrt(u**2 + v**2)
            under_35 = np.sum(speed < 35)
            total = len(speed)
            print(f"Total entries: {total}")
            print(f"Entries with velocity < 35 m/s: {under_35} ({under_35/total*100:.1f}%)")
            
            plot_uv_scatter(u, v, args.output)
        else:
            print("No valid cell data found (dx, dy, dt must be present and dt != 0).")
