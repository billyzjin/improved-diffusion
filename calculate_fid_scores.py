#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path
import numpy as np

# This is a workaround to import the fid score calculation functions directly.
# We will add the library path to the system path.
try:
    import pytorch_fid.fid_score
except ImportError:
    # Get the site-packages directory
    result = subprocess.run([sys.executable, '-m', 'site', '--user-site'], capture_output=True, text=True)
    site_packages = result.stdout.strip()
    if site_packages and os.path.exists(site_packages):
        sys.path.append(site_packages)
        print(f"Added {site_packages} to system path.", file=sys.stderr)
    try:
        import pytorch_fid.fid_score
    except ImportError:
        print("ERROR: pytorch-fid is not installed correctly. Please run 'pip install pytorch-fid'.", file=sys.stderr)
        sys.exit(1)


def main():
    """
    Calculates FID scores for a directory of experiments.
    """
    # 1. Find the latest evaluation directory from the command line argument.
    if len(sys.argv) != 2:
        print("Usage: python calculate_fid_scores.py <path_to_parent_evaluation_directory>")
        sys.exit(1)
    
    latest_eval_dir = Path(sys.argv[1])
    if not latest_eval_dir.is_dir():
        print(f"ERROR: Directory not found: {latest_eval_dir}")
        sys.exit(1)

    print(f"Calculating FID for results located in: {latest_eval_dir}")

    # 2. Define paths
    device = "cuda"
    cifar_train_path = Path("./cifar_train")
    cifar_stats_file = Path("/scratch/bjin0/cifar10_train_stats.npz")
    fid_results_file = latest_eval_dir / "fid_scores.txt"

    if not cifar_train_path.is_dir():
        print(f"ERROR: Real dataset not found at {cifar_train_path}")
        print("Please ensure the 'cifar_train' directory exists in the current folder.")
        sys.exit(1)

    # 3. Pre-calculate statistics for the real CIFAR-10 dataset if they don't exist.
    if not cifar_stats_file.exists():
        print(f"\n--- Pre-calculating statistics for the real CIFAR-10 dataset ---")
        print(f"This is a one-time operation. Statistics will be saved to {cifar_stats_file}")
        
        # We call the internal function, which gives us more control.
        # CORRECTED: The output path is the second positional argument, not a keyword argument.
        pytorch_fid.fid_score.save_fid_stats(
            [str(cifar_train_path)],
            str(cifar_stats_file),
            batch_size=50,
            device=device,
            dims=2048,
            num_workers=4
        )
        print("--- Statistics calculated and saved successfully. ---")
    else:
        print(f"\n--- Found pre-calculated CIFAR-10 statistics at {cifar_stats_file}. Skipping calculation. ---")

    # 4. Find all experiment subdirectories
    experiments = sorted([d for d in latest_eval_dir.iterdir() if d.is_dir()])
    print(f"\nFound {len(experiments)} experiment directories to evaluate.")

    # 5. Calculate FID for each experiment and save to a file.
    with open(fid_results_file, "w") as f_out:
        print("\n--- Calculating FID Scores for Generated Samples ---")
        f_out.write("FID SCORES (lower is better):\n")
        f_out.write("=============================\n")
        
        for exp_dir in experiments:
            sample_file = exp_dir / f"samples_50000x32x32x3.npz"
            print(f"\nProcessing: {exp_dir.name}")
            
            if sample_file.exists():
                print("    Calculating FID score...")
                
                fid_value = pytorch_fid.fid_score.calculate_fid_given_paths(
                    paths=[str(sample_file), str(cifar_stats_file)],
                    batch_size=50,
                    device=device,
                    dims=2048,
                    num_workers=4,
                )
                
                print(f"    Done. FID score for {exp_dir.name}: {fid_value:.4f}")
                f_out.write(f"{exp_dir.name:<25}: {fid_value:.4f}\n")
            else:
                print(f"    WARNING: Sample file not found at {sample_file}")
                f_out.write(f"{exp_dir.name:<25}: SAMPLES NOT FOUND\n")

    print("\n==========================================")
    print("FID CALCULATION COMPLETE!")
    print(f"Results summary saved to: {fid_results_file}")
    print("==========================================")
    
    # Print the final results to the console for convenience
    print("\n--- Final FID Results ---")
    with open(fid_results_file, 'r') as f:
        print(f.read())


if __name__ == "__main__":
    main()
