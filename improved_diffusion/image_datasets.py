import os
import io
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image


def _is_lmdb_dir(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "data.mdb"))


def _center_crop_resize(image, image_size):
    image = image.convert("RGB")
    width, height = image.size
    scale = image_size / min(width, height)
    resample = getattr(Image, "Resampling", Image).BOX
    image = image.resize((int(round(scale * width)), int(round(scale * height))), resample=resample)
    arr = np.array(image)
    h, w, _ = arr.shape
    h_off = (h - image_size) // 2
    w_off = (w - image_size) // 2
    return arr[h_off : h_off + image_size, w_off : w_off + image_size]


def _normalize_uint8_hwc(arr):
    image = arr.astype(np.float32) / 255.0
    image = (image - 0.5) * 2.0
    return torch.from_numpy(image).permute(2, 0, 1)


class LmdbImageDataset(Dataset):
    def __init__(self, data_dir, image_size, class_cond=False):
        if class_cond:
            raise ValueError("class_cond=True is not supported for unlabeled LSUN LMDB datasets")
        self.data_dir = data_dir
        self.image_size = image_size
        self.class_cond = class_cond
        self._env = None

        env = self._open_env()
        with env.begin(write=False) as transaction:
            self.keys = list(transaction.cursor().iternext(keys=True, values=False))
        env.close()
        self._env = None
        print(f"Found {len(self.keys)} LMDB images in {data_dir}")

    def _open_env(self):
        if self._env is None:
            import lmdb

            self._env = lmdb.open(
                self.data_dir,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=32,
            )
        return self._env

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        env = self._open_env()
        key = self.keys[idx]
        with env.begin(write=False) as transaction:
            image_data = transaction.get(key)
        if image_data is None:
            raise KeyError(f"LMDB key disappeared: {key!r}")
        image = Image.open(io.BytesIO(image_data))
        arr = _center_crop_resize(image, self.image_size)
        return _normalize_uint8_hwc(arr), {}

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
        image = _normalize_uint8_hwc(np.array(image))
        
        if self.class_cond:
            return image, {"y": torch.tensor(int(self.labels[idx]), dtype=torch.long)}
        else:
            return image, {}

def make_dataset(data_dir, image_size, class_cond=False):
    if _is_lmdb_dir(data_dir):
        return LmdbImageDataset(data_dir, image_size, class_cond)
    return ImageDataset(data_dir, image_size, class_cond)


def load_data(data_dir, batch_size, image_size, class_cond=False, deterministic=False):
    dataset = make_dataset(data_dir, image_size, class_cond)
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
