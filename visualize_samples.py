#!/usr/bin/env python3

import numpy as np
from PIL import Image
import argparse
from pathlib import Path

def generate_grid_for_file(npz_path, grid_size):
    """
    Loads a single .npz file and saves a visualization grid.
    """
    output_path = npz_path.with_suffix(".png")
    num_samples_to_show = grid_size * grid_size

    print(f"Processing: {npz_path.name}")
    try:
        data = np.load(npz_path)
        samples = data['arr_0']
        data.close()
    except Exception as e:
        print(f"--> ERROR: Could not load NPZ file: {e}")
        return
        
    if len(samples) < num_samples_to_show:
        print(f"--> WARNING: Only {len(samples)} samples available. Skipping.")
        return

    # Get image dimensions from the first sample
    if samples.ndim != 4:
        print(f"--> ERROR: Samples have incorrect dimensions {samples.shape}. Skipping.")
        return
    img_height, img_width = samples.shape[1], samples.shape[2]
    
    # Create a blank canvas for the grid
    grid_img = Image.new('RGB', (img_width * grid_size, img_height * grid_size))
    
    # Paste each sample into the grid
    for i in range(num_samples_to_show):
        row, col = i // grid_size, i % grid_size
        try:
            img = Image.fromarray(samples[i], 'RGB')
            grid_img.paste(img, (col * img_width, row * img_height))
        except ValueError:
            print(f"--> ERROR: Could not convert sample #{i} to image. Skipping file.")
            return
        
    print(f"--> Saving image grid to {output_path}")
    grid_img.save(output_path)

def main():
    """
    Finds all .npz files in a directory and generates a PNG visualization grid for each.
    """
    parser = argparse.ArgumentParser(description="Visualize all .npz sample files in a directory.")
    parser.add_argument("directory_path", type=str, help="Path to the directory containing .npz sample files.")
    parser.add_argument("--grid_size", type=int, default=8, help="Number of images per row/column in the grid (e.g., 8 for an 8x8 grid).")
    args = parser.parse_args()

    directory_path = Path(args.directory_path)
    if not directory_path.is_dir():
        print(f"Error: Directory not found at {directory_path}")
        return

    # Find all .npz files in the directory
    npz_files = sorted(list(directory_path.glob("*.npz")))
    
    if not npz_files:
        print(f"No .npz files found in {directory_path}")
        return
        
    print(f"Found {len(npz_files)} .npz files to visualize.")
    print("==========================================")
    
    for npz_file in npz_files:
        generate_grid_for_file(npz_file, args.grid_size)

    print("==========================================")
    print("Visualization complete.")

if __name__ == "__main__":
    main()
