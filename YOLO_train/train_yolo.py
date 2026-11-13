"""Train the single-class Hexbug detector.

Run from the YOLO_train/ directory (data.yaml uses paths relative to it):

    python train_yolo.py                      # fine-tune from COCO yolov8s
    python train_yolo.py --weights runs/train/hexbug_train10/weights/last.pt --epochs 20
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="yolov8s.pt", help="starting checkpoint (auto-downloaded if a stock name)")
    ap.add_argument("--data", default=str(HERE / "data.yaml"))
    ap.add_argument("--epochs", default=80, type=int)
    ap.add_argument("--imgsz", default=640, type=int)
    ap.add_argument("--batch", default=16, type=int)
    ap.add_argument("--name", default="hexbug_train")
    ap.add_argument("--device", default=0 if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=10,              # early stop if no val improvement for 10 epochs
        name=args.name,
        project=str(HERE / "runs" / "train"),
        device=args.device,
        cos_lr=True,
        lr0=0.005,
        close_mosaic=10,
    )


if __name__ == "__main__":
    main()
