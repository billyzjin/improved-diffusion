"""
Image datasets - modified to work without MPI.
"""
import os
import torch as th
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, data_dir, image_size, class_cond=False):
        self.data_dir = data_dir
        self.image_size = image_size
        self.class_cond = class_cond

        self.image_paths = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))

        print(f"Found {len(self.image_paths)} images in {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        image = Image.open(path).convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = th.from_numpy(image).permute(2, 0, 1)

        if self.class_cond:
            class_name = os.path.basename(path).split('_')[0]
            class_id = hash(class_name) % 1000
            return image, class_id
        else:
            return image, 0

def load_data(data_dir, batch_size, image_size, class_cond=False):
    """Load image dataset."""
    dataset = ImageDataset(data_dir, image_size, class_cond)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    # Convert to infinite iterator and fix format
    def infinite_dataloader():
        while True:
            for batch_data in dataloader:
                if class_cond:
                    images, labels = batch_data
                    cond = {"y": labels} if class_cond else {}
                    yield images, cond
                else:
                    images, _ = batch_data
                    cond = {}
                    yield images, cond
    return infinite_dataloader()
