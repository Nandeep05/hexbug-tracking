"""Convert CSV point annotations into YOLO label files.

For every (video, frame) in dataset/csv_files/*.csv, writes
YOLO_train/labels/train/<video>_frame<NNNN>.txt containing one line per Hexbug:

    0 x_center y_center width height      (all normalised to [0, 1])

Each Hexbug is labelled with a fixed BBOX_SIZE x BBOX_SIZE pixel box centred on
its annotated head position. Rows are grouped by frame so multi-Hexbug frames
keep every annotation (an earlier version overwrote the file per row and lost them).
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BBOX_SIZE = 20
CLASS_ID = 0


def frame_size(images_dir: Path, base: str):
    """Width/height of the first extracted frame of this video (all frames share it)."""
    for p in sorted(images_dir.glob(f"{base}_frame*.png")):
        img = cv2.imread(str(p))
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=ROOT / "YOLO_train" / "images" / "train", type=Path)
    ap.add_argument("--csvs", default=ROOT / "dataset" / "csv_files", type=Path)
    ap.add_argument("--out", default=ROOT / "YOLO_train" / "labels" / "train", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    total_frames = total_boxes = 0

    for csv_path in sorted(args.csvs.glob("*.csv")):
        base = csv_path.stem
        size = frame_size(args.images, base)
        if size is None:
            print(f"[!] No extracted frames for {base}; run scripts/Generate_bboxes.py first")
            continue
        w, h = size
        w_norm, h_norm = BBOX_SIZE / w, BBOX_SIZE / h

        df = pd.read_csv(csv_path)
        for frame_num, rows in df.groupby("t"):
            frame_num = int(frame_num)
            if not (args.images / f"{base}_frame{frame_num:04d}.png").exists():
                print(f"[!] Image not found: {base}_frame{frame_num:04d}.png")
                continue

            lines = [
                f"{CLASS_ID} {float(r.x) / w:.6f} {float(r.y) / h:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                for r in rows.itertuples()
            ]
            (args.out / f"{base}_frame{frame_num:04d}.txt").write_text("".join(lines))
            total_frames += 1
            total_boxes += len(lines)

        print(f"[✓] {base}: {len(df)} annotations over {df['t'].nunique()} frames")

    print(f"\nWrote {total_frames} label files, {total_boxes} boxes total")


if __name__ == "__main__":
    main()
