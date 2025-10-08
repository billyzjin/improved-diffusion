#!/usr/bin/env python3
"""
Manual CIFAR-10 dataset preparation that doesn't require network access.
This creates dummy CIFAR-10 data for testing purposes.
"""

import os
import numpy as np
from PIL import Image

CLASSES = (
    "plane", "car", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
)

def create_dummy_cifar10():
    """Create dummy CIFAR-10 dataset for testing."""
    
    for split in ["train", "test"]:
        out_dir = f"cifar_{split}"
        if os.path.exists(out_dir):
            print(f"skipping split {split} since {out_dir} already exists.")
            continue
            
        print(f"Creating dummy {split} dataset...")
        os.mkdir(out_dir)
        
        # Create 1000 dummy images for training, 200 for test
        num_images = 1000 if split == "train" else 200
        
        for i in range(num_images):
            # Create random 32x32 RGB image
            random_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            img = Image.fromarray(random_image)
            
            # Random class label
            label = i % len(CLASSES)
            filename = os.path.join(out_dir, f"{CLASSES[label]}_{i:05d}.png")
            img.save(filename)
            
        print(f"Created {num_images} dummy images in {out_dir}")

if __name__ == "__main__":
    create_dummy_cifar10()
