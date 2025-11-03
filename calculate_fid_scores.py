#!/usr/bin/env python3

import os
import sys
import subprocess
import tempfile
import numpy as np
from PIL import Image
from pathlib import Path
import shutil

def main():
    """
    Calculates FID scores for a directory of experiments by extracting NPZ samples
    to a temporary image directory.
    """
    # 1. Check for command-line argument for the evaluation directory
    if len(sys.argv) != 2:
        print("Usage: python calculate_fid_scores.py <path_to_parent_evaluation_directory>")
        sys.exit(1)
    
    latest_eval_dir = Path(sys.argv[1])
    if not latest_eval_dir.is_dir():
        print(f"ERROR: Directory not found: {latest_eval_dir}")
        sys.exit(1)

    print(f"Calculating FID for results located in: {latest_eval_dir}")

    # 2. Define essential paths
    device = "cuda"
    cifar_train_path = Path("./cifar_train")
    cifar_stats_file = Path("/scratch/bjin0/cifar10_train_stats.npz")
    fid_results_file = latest_eval_dir / "fid_scores.txt"

    if not cifar_train_path.is_dir():
        print(f"ERROR: Real dataset not found at {cifar_train_path}")
        sys.exit(1)

    # 3. Pre-calculate statistics for the real CIFAR-10 dataset if they don't exist
    if not cifar_stats_file.exists():
        print(f"\n--- Pre-calculating statistics for the real CIFAR-10 dataset ---")
        # The command-line tool is the most stable way to do this.
        subprocess.run(
            [
                sys.executable, "-m", "pytorch_fid",
                str(cifar_train_path),
                str(cifar_stats_file),
                "--save-stats",
                "--device", device,
            ],
            check=True,
        )
        print("--- Statistics calculated successfully. ---")
    else:
        print(f"\n--- Found pre-calculated CIFAR-10 statistics. ---")

    # 4. Find experiment subdirectories
    experiments = sorted([d for d in latest_eval_dir.iterdir() if d.is_dir()])
    
    # 5. Create a temporary directory in scratch space for the images
    with tempfile.TemporaryDirectory(dir="/scratch/bjin0", prefix="fid_temp_images_") as temp_img_dir:
        temp_img_path = Path(temp_img_dir)
        print(f"\nCreated temporary directory for images: {temp_img_path}")

        with open(fid_results_file, "w") as f_out:
            f_out.write("FID SCORES (lower is better):\n=============================\n")
            
            for exp_dir in experiments:
                sample_file = exp_dir / "samples_50000x32x32x3.npz"
                print(f"\nProcessing: {exp_dir.name}")

                if not sample_file.exists():
                    print(f"    WARNING: Sample file not found.")
                    f_out.write(f"{exp_dir.name:<25}: SAMPLES NOT FOUND\n")
                    continue

                # A. Load the NPZ file
                print(f"    Loading samples from {sample_file}...")
                try:
                    data = np.load(sample_file)
                    # The key for the image data is typically 'arr_0'
                    images = data['arr_0']
                    data.close()
                except Exception as e:
                    print(f"    ERROR: Failed to load or read NPZ file: {e}")
                    f_out.write(f"{exp_dir.name:<25}: FAILED TO LOAD NPZ\n")
                    continue
                
                # B. Save images to the temporary directory
                print(f"    Saving {len(images)} images to temporary directory...")
                for i, img_array in enumerate(images):
                    img = Image.fromarray(img_array, 'RGB')
                    img.save(temp_img_path / f"sample_{i:05d}.png")
                
                # C. Calculate FID using the directory of images
                print(f"    Calculating FID score...")
                result = subprocess.run(
                    [
                        sys.executable, "-m", "pytorch_fid",
                        str(temp_img_path),
                        str(cifar_stats_file),
                        "--device", device,
                    ],
                    capture_output=True, text=True, check=True
                )
                fid_score = result.stdout.strip()
                
                print(f"    Done. FID score for {exp_dir.name}: {fid_score}")
                f_out.write(f"{exp_dir.name:<25}: {fid_score}\n")
                
                # D. Clean up images for the next run
                print("    Cleaning up temporary images...")
                for item in temp_img_path.glob('*'):
                    item.unlink()

    print("\n==========================================")
    print("FID CALCULATION COMPLETE!")
    print(f"Results summary saved to: {fid_results_file}")
    print("==========================================")
    
    print("\n--- Final FID Results ---")
    with open(fid_results_file, 'r') as f:
        print(f.read())

if __name__ == "__main__":
    main()
