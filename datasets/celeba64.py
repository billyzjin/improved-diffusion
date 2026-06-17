import argparse
import shutil
from pathlib import Path

from PIL import Image


EXPECTED_COUNTS = {
    "train": 162770,
    "valid": 19867,
    "test": 19962,
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


def center_crop_square(image):
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def prepare_split(split, root, out_root, overwrite, size):
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

    print(f"loading CelebA split={split} from {root}")
    try:
        import torchvision

        dataset = torchvision.datasets.CelebA(root=str(root), split=split, target_type="attr", download=True)
    except Exception as exc:
        raise SystemExit(
            "Unable to load/download CelebA via torchvision. If automatic download is blocked, "
            "download the CelebA aligned images and metadata into --root using torchvision's "
            "expected layout, then rerun with the same --root.\n"
            f"Original error: {exc}"
        ) from exc

    if len(dataset) != expected:
        raise SystemExit(f"split {split}: torchvision returned {len(dataset)} images, expected {expected}")

    print(f"dumping {len(dataset)} center-cropped {size}x{size} images to {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    resample = getattr(Image, "Resampling", Image).LANCZOS
    for i in progress(range(len(dataset))):
        item = dataset[i]
        image = item[0] if isinstance(item, tuple) else item
        image = center_crop_square(image.convert("RGB")).resize((size, size), resample)
        image.save(out_dir / f"celeba_{split}_{i:06d}.png")

    written = count_pngs(out_dir)
    if written != expected:
        raise SystemExit(f"split {split}: wrote {written} PNGs, expected {expected}")
    print(f"split {split}: wrote {written} PNGs")


def main():
    parser = argparse.ArgumentParser(description="Convert aligned CelebA to center-cropped 64x64 PNG folders.")
    parser.add_argument(
        "--root",
        default="/project_gpfs/bata0/bjin0/celeba_64x64/source",
        help="Directory for torchvision CelebA downloads/source files.",
    )
    parser.add_argument(
        "--out_root",
        default="/project_gpfs/bata0/bjin0/celeba_64x64",
        help="Output root containing train/, valid/, and test/.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=sorted(EXPECTED_COUNTS),
        help="CelebA splits to convert.",
    )
    parser.add_argument("--size", type=int, default=64, help="Output image size.")
    parser.add_argument("--overwrite", action="store_true", help="Remove and rebuild existing split directories.")
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        prepare_split(split, root, out_root, args.overwrite, args.size)


if __name__ == "__main__":
    main()
