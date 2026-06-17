import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from PIL import Image


CLASSES = (
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
)

EXPECTED_COUNTS = {
    "train": 50000,
    "test": 10000,
}


def progress(iterable):
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable)
    except ImportError:
        return iterable


def count_pngs(path):
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*.png"))


def image_hash(path):
    image = Image.open(path).convert("RGB")
    h = hashlib.sha256()
    h.update(image.size[0].to_bytes(4, "little"))
    h.update(image.size[1].to_bytes(4, "little"))
    h.update(image.tobytes())
    return h.hexdigest()


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

    print(f"downloading/loading CIFAR-100 split={split} from {root}")
    import torchvision

    dataset = torchvision.datasets.CIFAR100(root=str(root), train=split == "train", download=True)
    if len(dataset) != expected:
        raise SystemExit(f"split {split}: torchvision returned {len(dataset)} images, expected {expected}")

    print(f"dumping {len(dataset)} images to {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    for i in progress(range(len(dataset))):
        image, label = dataset[i]
        label = int(label)
        filename = out_dir / f"{CLASSES[label]}_{i:05d}.png"
        image.save(filename)

    written = count_pngs(out_dir)
    if written != expected:
        raise SystemExit(f"split {split}: wrote {written} PNGs, expected {expected}")
    print(f"split {split}: wrote {written} PNGs")


def write_overlap_report(out_root, compare_dirs):
    report_path = out_root / "cifar10_overlap_report.tsv"
    rows = []
    cifar100_hashes = {}
    for split in ("train", "test"):
        split_dir = out_root / split
        for path in split_dir.rglob("*.png"):
            cifar100_hashes[image_hash(path)] = str(path)

    for compare_dir in compare_dirs:
        compare_path = Path(compare_dir)
        if not compare_path.is_dir():
            rows.append((str(compare_path), "missing_dir", "", ""))
            continue
        overlap = 0
        first_match = ""
        first_source = ""
        for path in compare_path.rglob("*.png"):
            digest = image_hash(path)
            if digest in cifar100_hashes:
                overlap += 1
                if not first_match:
                    first_match = str(path)
                    first_source = cifar100_hashes[digest]
        rows.append((str(compare_path), str(overlap), first_match, first_source))

    with report_path.open("w") as f:
        f.write("compare_dir\texact_png_hash_overlap\tfirst_compare_match\tfirst_cifar100_match\n")
        for row in rows:
            f.write("\t".join(row) + "\n")
    print(f"wrote overlap report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert torchvision CIFAR-100 to PNG folders.")
    parser.add_argument(
        "--root",
        default="/project_gpfs/bata0/bjin0/cifar100_32x32/source",
        help="Directory for torchvision CIFAR-100 downloads.",
    )
    parser.add_argument(
        "--out_root",
        default="/project_gpfs/bata0/bjin0/cifar100_32x32",
        help="Output root containing train/ and test/.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=sorted(EXPECTED_COUNTS),
        help="CIFAR-100 splits to convert.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove and rebuild existing split directories.")
    parser.add_argument(
        "--compare_cifar10_dirs",
        nargs="*",
        default=["cifar_train", "cifar_test"],
        help="Optional CIFAR-10 directories for exact PNG hash overlap reporting.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=str(root.parent)) as tmp:
        download_root = root if root.exists() else Path(tmp)
        download_root.mkdir(parents=True, exist_ok=True)
        for split in args.splits:
            dump_split(split, download_root, out_root, args.overwrite)

    write_overlap_report(out_root, args.compare_cifar10_dirs)


if __name__ == "__main__":
    main()
