import os
import tempfile

import torchvision
from PIL import Image
from tqdm.auto import tqdm

# Use filesystem-safe class names (FashionMNIST label order 0..9).
CLASSES = (
    "tshirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankleboot",
)


def main():
    for split in ["train", "test"]:
        out_dir = f"fashion_{split}"
        if os.path.exists(out_dir):
            print(f"skipping split {split} since {out_dir} already exists.")
            continue

        print("downloading...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = torchvision.datasets.FashionMNIST(
                root=tmp_dir, train=split == "train", download=True
            )

        print("dumping images...")
        os.mkdir(out_dir)
        for i in tqdm(range(len(dataset))):
            image, label = dataset[i]  # PIL image in mode "L"
            # Match CIFAR-10 pipeline expectations: RGB, 32x32.
            image = image.convert("RGB")
            image = image.resize((32, 32), resample=Image.BICUBIC)
            filename = os.path.join(out_dir, f"{CLASSES[label]}_{i:05d}.png")
            image.save(filename)


if __name__ == "__main__":
    main()


