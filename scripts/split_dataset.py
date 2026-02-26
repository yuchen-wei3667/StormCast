#!/usr/bin/env python3
import os
import json
import random

def main():
    data_dir = "/home/yuchenwei/StormCast_Data"
    all_files = []
    
    # Track discovery
    for date_folder in sorted(os.listdir(data_dir)):
        date_path = os.path.join(data_dir, date_folder)
        if not os.path.isdir(date_path): continue
        for json_file in os.listdir(date_path):
            if json_file.endswith('.json') and json_file != 'cell_index.json':
                all_files.append(os.path.join(date_path, json_file))
    
    print(f"Found {len(all_files)} total tracks.")
    
    random.seed(42)
    random.shuffle(all_files)
    
    split_idx = int(0.8 * len(all_files))
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]
    
    os.makedirs('data_splits', exist_ok=True)
    
    with open('data_splits/train_files.json', 'w') as f:
        json.dump(train_files, f, indent=2)
        
    with open('data_splits/test_files.json', 'w') as f:
        json.dump(test_files, f, indent=2)
        
    print(f"Dataset split complete:")
    print(f"  Train: {len(train_files)} tracks")
    print(f"  Test:  {len(test_files)} tracks")
    print(f"List saved to data_splits/train_files.json and data_splits/test_files.json")

if __name__ == "__main__":
    main()
