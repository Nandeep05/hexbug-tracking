"""Run the trained detector on a video and save an annotated copy (detection only, no tracking).

    python test_model.py --video ../dataset/test_videos/test005.mp4
"""
import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "YOLO_train" / "runs" / "train" / "hexbug_train10" / "weights" / "best.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, type=Path)
    ap.add_argument("--video", default=ROOT / "dataset" / "test_videos" / "test005.mp4", type=Path)
    ap.add_argument("--out", default=None, type=Path, help="default: YOLO_train/ran_tests/<video>_detect.mp4")
    ap.add_argument("--conf", default=0.2, type=float)
    ap.add_argument("--device", default=0 if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_path = args.out or ROOT / "YOLO_train" / "ran_tests" / f"{args.video.stem}_detect.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    cap = cv2.VideoCapture(str(args.video))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=args.conf, iou=0.45, device=args.device, verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"Head ({conf:.2f})", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame)

    cap.release()
    out.release()
    print(f"[INFO] Detection video saved to: {out_path}")


if __name__ == "__main__":
    main()
