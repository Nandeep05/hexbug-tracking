"""Split the dataset into train / val BY VIDEO.

Whole videos go to validation, never individual frames. Adjacent frames of one
video are near-duplicates, so a per-frame split leaks training data into
validation and inflates the metrics.

The script is idempotent: it first moves everything in val/ back to train/, then
re-splits with a fixed seed, so running it twice gives the same result.
"""
import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def move_all(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_file():
            shutil.move(str(p), str(dst / p.name))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT / "YOLO_train", type=Path)
    ap.add_argument("--val-ratio", default=0.2, type=float)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    img_train, img_val = args.root / "images" / "train", args.root / "images" / "val"
    lbl_train, lbl_val = args.root / "labels" / "train", args.root / "labels" / "val"

    # Reset: everything back to train so the split is reproducible.
    if img_val.exists():
        move_all(img_val, img_train)
    if lbl_val.exists():
        move_all(lbl_val, lbl_train)
    img_val.mkdir(parents=True, exist_ok=True)
    lbl_val.mkdir(parents=True, exist_ok=True)

    images = [p for p in img_train.iterdir() if p.suffix.lower() in (".png", ".jpg")]
    videos = sorted({p.name.split("_frame")[0] for p in images})

    rng = random.Random(args.seed)
    rng.shuffle(videos)
    n_val = max(1, int(len(videos) * args.val_ratio))
    val_videos = set(videos[:n_val])

    moved = 0
    for img in images:
        if img.name.split("_frame")[0] not in val_videos:
            continue
        shutil.move(str(img), str(img_val / img.name))
        lbl = lbl_train / f"{img.stem}.txt"
        if lbl.exists():
            shutil.move(str(lbl), str(lbl_val / lbl.name))
        moved += 1

    print(f"Videos: {len(videos)} total, {n_val} in val: {sorted(val_videos)}")
    print(f"Images: {len(images)} total, {moved} moved to val")
    print("[✓] Train/validation split complete.")


if __name__ == "__main__":
    main()
