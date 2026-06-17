import argparse
import os
import shutil
from pathlib import Path

import torchvision
from tqdm.auto import tqdm


CLASSES = tuple(f"digit{i}" for i in range(10))
EXPECTED_COUNTS = {
    "train": 73257,
    "test": 26032,
    "extra": 531131,
}


def count_pngs(path):
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*.png"))


def dump_split(split, root, out_root, overwrite):
    out_dir = out_root / split
    expected = EXPECTED_COUNTS[split]

    if out_dir.exists():
        existing = count_pngs(out_dir)
        if existing >= expected and not overwrite:
            print(f"skipping split {split}; found {existing} PNGs in {out_dir}")
            return
        if not overwrite:
            raise SystemExit(
                f"{out_dir} exists but has {existing} PNGs; expected {expected}. "
                "Pass --overwrite to rebuild it."
            )
        shutil.rmtree(out_dir)

    print(f"downloading/loading SVHN split={split} from {root}")
    dataset = torchvision.datasets.SVHN(root=str(root), split=split, download=True)

    print(f"dumping {len(dataset)} images to {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    for i in tqdm(range(len(dataset))):
        image, label = dataset[i]
        label = int(label)
        filename = out_dir / f"{CLASSES[label]}_{i:06d}.png"
        image.save(filename)

    written = count_pngs(out_dir)
    if written != expected:
        raise SystemExit(f"split {split}: wrote {written} PNGs, expected {expected}")
    print(f"split {split}: wrote {written} PNGs")


def main():
    parser = argparse.ArgumentParser(description="Convert torchvision SVHN to PNG folders.")
    parser.add_argument(
        "--root",
        default="/project_gpfs/bata0/bjin0/svhn_32x32/source",
        help="Directory for torchvision .mat downloads.",
    )
    parser.add_argument(
        "--out_root",
        default="/project_gpfs/bata0/bjin0/svhn_32x32",
        help="Output root containing train/, test/, and optionally extra/.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=sorted(EXPECTED_COUNTS),
        help="SVHN splits to convert.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove and rebuild existing split directories.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        dump_split(split, root, out_root, args.overwrite)


if __name__ == "__main__":
    main()
