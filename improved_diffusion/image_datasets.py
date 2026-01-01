import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, data_dir, image_size, class_cond=False):
        self.data_dir = data_dir
        self.image_size = image_size
        self.class_cond = class_cond
        
        # Get list of image files
        self.image_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, file))
        
        print(f"Found {len(self.image_files)} images in {data_dir}")
        
        # If class conditional, derive stable labels from filename prefixes.
        # Expected filename pattern: "<class>_XXXX.png" (as described in README).
        self.class_to_id = None
        self.labels = None
        if class_cond:
            class_names = []
            for p in self.image_files:
                base = os.path.basename(p)
                class_names.append(base.split("_")[0] if "_" in base else base)
            unique = sorted(set(class_names))
            self.class_to_id = {name: i for i, name in enumerate(unique)}
            self.labels = np.array([self.class_to_id[n] for n in class_names], dtype=np.int64)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - 0.5) * 2.0  # Normalize to [-1, 1]
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        
        if self.class_cond:
            return image, {"y": torch.tensor(int(self.labels[idx]), dtype=torch.long)}
        else:
            return image, {}

def load_data(data_dir, batch_size, image_size, class_cond=False, deterministic=False):
    dataset = ImageDataset(data_dir, image_size, class_cond)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader
