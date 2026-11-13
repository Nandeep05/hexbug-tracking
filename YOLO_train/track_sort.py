"""Detect with the trained YOLO model and track with SORT (Kalman + IoU + Hungarian).

This is the supported tracking path.

    python track_sort.py --video ../dataset/test_videos/test001.mp4
"""
import argparse
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoid duplicate OpenMP runtime abort on Windows

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from sort.sort import Sort

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "YOLO_train" / "runs" / "train" / "hexbug_train10" / "weights" / "best.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, type=Path)
    ap.add_argument("--video", default=ROOT / "dataset" / "test_videos" / "test001.mp4", type=Path)
    ap.add_argument("--out", default=None, type=Path, help="default: YOLO_train/ran_tests/<video>_sort.mp4")
    ap.add_argument("--conf", default=0.2, type=float)
    ap.add_argument("--device", default=0 if torch.cuda.is_available() else "cpu")
    ap.add_argument("--show-detections", action="store_true", help="also draw raw YOLO boxes in blue")
    args = ap.parse_args()

    out_path = args.out or ROOT / "YOLO_train" / "ran_tests" / f"{args.video.stem}_sort.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    cap = cv2.VideoCapture(str(args.video))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.3)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=args.conf, iou=0.45, device=args.device, verbose=False)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append([x1, y1, x2, y2, conf])
            if args.show_detections:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f"YOLO {conf:.2f}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        detections = np.array(detections) if detections else np.empty((0, 5))
        for x1, y1, x2, y2, track_id in tracker.update(detections):  # SORT returns [x1, y1, x2, y2, id]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(width, int(x2)), min(height, int(y2))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {int(track_id)}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"[INFO] Tracked {frame_count} frames -> {out_path}")


if __name__ == "__main__":
    main()
