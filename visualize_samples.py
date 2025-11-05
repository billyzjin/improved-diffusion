#!/usr/bin/env python3

import numpy as np
from PIL import Image
import argparse
from pathlib import Path

def main():
    """
    Loads samples from a .npz file and saves them as a PNG image grid.
    """
    parser = argparse.ArgumentParser(description="Visualize samples from a .npz file.")
    parser.add_argument("npz_path", type=str, help="Path to the .npz sample file.")
    parser.add_argument("--output_path", type=str, help="Path to save the output PNG image. Defaults to the same name as the input.")
    parser.add_argument("--grid_size", type=int, default=8, help="Number of images per row/column in the grid (e.g., 8 for an 8x8 grid).")
    args = parser.parse_args()

    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        print(f"Error: File not found at {npz_path}")
        return

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = npz_path.with_suffix(".png")
    
    grid_size = args.grid_size
    num_samples_to_show = grid_size * grid_size

    print(f"Loading samples from {npz_path}...")
    try:
        data = np.load(npz_path)
        samples = data['arr_0']
        data.close()
    except Exception as e:
        print(f"Error loading NPZ file: {e}")
        return
        
    if len(samples) < num_samples_to_show:
        print(f"Warning: Only {len(samples)} samples available. Need {num_samples_to_show} for a {grid_size}x{grid_size} grid.")
        return

    print(f"Creating a {grid_size}x{grid_size} image grid...")
    
    # Get image dimensions from the first sample
    _, height, width, channels = samples.shape
    
    # Create a blank canvas for the grid
    grid_img = Image.new('RGB', (width * grid_size, height * grid_size))
    
    # Paste each sample into the grid
    for i in range(num_samples_to_show):
        row = i // grid_size
        col = i % grid_size
        img = Image.fromarray(samples[i], 'RGB')
        grid_img.paste(img, (col * width, row * height))
        
    print(f"Saving image grid to {output_path}...")
    grid_img.save(output_path)
    print("Done.")

if __name__ == "__main__":
    main()
