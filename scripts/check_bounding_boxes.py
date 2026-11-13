"""QA tool: overlay the raw annotations (CSV or .traco) on every video frame.

Output goes to scripts/debug_frames/<video>/ for eyeballing; nothing here is
used for training.

    python check_bounding_boxes.py            # uses .traco files
    python check_bounding_boxes.py --csv      # uses CSV files
"""
import argparse
import json
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_traco(path: Path):
    annotations = {}
    for entry in json.loads(path.read_text())["rois"]:
        annotations.setdefault(entry["z"], []).append((entry["id"], *entry["pos"]))
    return annotations


def load_csv(path: Path):
    df = pd.read_csv(path)
    return {int(t): [(int(r.hexbug), r.x, r.y) for r in g.itertuples()] for t, g in df.groupby("t")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=ROOT / "dataset" / "raw_videos", type=Path)
    ap.add_argument("--csvs", default=ROOT / "dataset" / "csv_files", type=Path)
    ap.add_argument("--tracos", default=ROOT / "dataset" / "txt_files", type=Path)
    ap.add_argument("--out", default=ROOT / "scripts" / "debug_frames", type=Path)
    ap.add_argument("--csv", action="store_true", help="use CSV annotations instead of .traco")
    args = ap.parse_args()

    for video_path in sorted(args.videos.glob("*.mp4")):
        name = video_path.stem
        ann_path = (args.csvs / f"{name}.csv") if args.csv else (args.tracos / f"{name}.traco")
        if not ann_path.is_file():
            print(f"[!] Skipping {name}: {ann_path.name} not found")
            continue

        try:
            annotations = load_csv(ann_path) if args.csv else load_traco(ann_path)
        except Exception as e:
            print(f"[!] Error reading {ann_path.name}: {e}")
            continue

        out_dir = args.out / name
        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            for bug_id, x, y in annotations.get(frame_num, []):
                if 0 <= int(x) < w and 0 <= int(y) < h:
                    cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), -1)
                    cv2.putText(frame, f"ID {bug_id}", (int(x) + 5, int(y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    print(f"[Warning] {name} frame {frame_num}: coordinate ({x},{y}) outside frame")
            cv2.imwrite(str(out_dir / f"{name}_frame_{frame_num:04d}.jpg"), frame)
            frame_num += 1

        cap.release()
        print(f"[✓] {name} ({frame_num} frames)")


if __name__ == "__main__":
    main()
