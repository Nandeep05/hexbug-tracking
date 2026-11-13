"""Extract clean training frames from the raw videos.

Writes every frame of every video in dataset/raw_videos to YOLO_train/images/train
as <video>_frame<NNNN>.png. Frames are saved UNMODIFIED so the detector learns the
Hexbug's appearance, not a drawn box.

Pass --draw to additionally write a QA copy with all annotated points boxed to
scripts/debug_frames/bboxes/. Those QA images are never used for training.
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BBOX_SIZE = 20


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=ROOT / "dataset" / "raw_videos", type=Path)
    ap.add_argument("--csvs", default=ROOT / "dataset" / "csv_files", type=Path)
    ap.add_argument("--out", default=ROOT / "YOLO_train" / "images" / "train", type=Path)
    ap.add_argument("--draw", action="store_true", help="also write boxed QA frames to scripts/debug_frames/bboxes")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    debug_dir = ROOT / "scripts" / "debug_frames" / "bboxes"
    if args.draw:
        debug_dir.mkdir(parents=True, exist_ok=True)

    half = BBOX_SIZE // 2

    for video_path in sorted(args.videos.glob("*.mp4")):
        base = video_path.stem
        csv_path = args.csvs / f"{base}.csv"
        if not csv_path.exists():
            print(f"[!] CSV not found for {base}, skipping")
            continue

        df = pd.read_csv(csv_path)
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            name = f"{base}_frame{frame_idx:04d}.png"
            cv2.imwrite(str(args.out / name), frame)  # clean frame

            if args.draw:
                qa = frame.copy()
                for _, row in df[df["t"] == frame_idx].iterrows():  # every hexbug in this frame
                    x, y = int(float(row["x"])), int(float(row["y"]))
                    cv2.rectangle(qa, (x - half, y - half), (x + half, y + half), (0, 255, 0), 2)
                    cv2.putText(qa, f"ID {int(row['hexbug'])}", (x + half + 2, y - half),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imwrite(str(debug_dir / name), qa)

            frame_idx += 1

        cap.release()
        print(f"[✓] {base}: {frame_idx} frames")


if __name__ == "__main__":
    main()
