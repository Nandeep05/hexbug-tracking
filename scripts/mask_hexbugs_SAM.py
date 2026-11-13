"""Annotation-assistance experiment: use SAM to turn point annotations into boxes.

For each annotated frame, prompts Segment Anything with the Hexbug head points
from the .traco file, derives a tight box from each mask, and saves a
visualisation to dataset/bounding_boxes/. This was explored as a way to get
real object extents instead of the fixed 20x20 boxes; it is NOT part of the
training pipeline. Requires the SAM ViT-B checkpoint in checkpoints/
(https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth).

    pip install git+https://github.com/facebookresearch/segment-anything.git
    python mask_hexbugs_SAM.py
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

ROOT = Path(__file__).resolve().parents[1]


def xywh_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()]


def draw_bbox(image, bbox, color=(0, 255, 0), label="hexbug"):
    x, y, w, h = map(int, bbox)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=ROOT / "dataset" / "raw_videos", type=Path)
    ap.add_argument("--tracos", default=ROOT / "dataset" / "txt_files", type=Path)
    ap.add_argument("--out", default=ROOT / "dataset" / "bounding_boxes", type=Path)
    ap.add_argument("--checkpoint", default=ROOT / "checkpoints" / "sam_vit_b_01ec64.pth", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint)).to(device)
    predictor = SamPredictor(sam)

    for video_path in sorted(args.videos.glob("*.mp4")):
        base = video_path.stem
        traco_path = args.tracos / f"{base}.traco"
        if not traco_path.exists():
            print(f"[!] Missing annotation for {base}")
            continue

        rois = json.loads(traco_path.read_text()).get("rois", [])
        if not rois:
            print(f"[!] No ROIs in {traco_path.name}")
            continue

        rois_by_frame = {}
        for roi in rois:
            rois_by_frame.setdefault(roi["z"], []).append(roi["pos"])

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\nProcessing {base} ({total_frames} frames)...")

        for frame_idx in sorted(rois_by_frame):
            if frame_idx >= total_frames:
                print(f"[!] Frame {frame_idx} out of range")
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"[!] Failed to read frame {frame_idx}")
                continue

            predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            points = np.array(rois_by_frame[frame_idx])

            bboxes = []
            for point in points:  # one prompt per hexbug so masks don't merge
                masks, _, _ = predictor.predict(
                    point_coords=np.array([point]), point_labels=np.array([1]), multimask_output=False
                )
                for mask in masks:
                    bbox = xywh_from_mask(mask)
                    if bbox:
                        bboxes.append(bbox)
                        draw_bbox(frame, bbox)

            for pt in points:
                cv2.circle(frame, tuple(map(int, pt)), 3, (0, 0, 255), -1)

            out_name = f"{base}_frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(args.out / out_name), frame)
            print(f"[✓] {out_name} ({len(bboxes)} boxes)")

        cap.release()

    print("\nDone.")


if __name__ == "__main__":
    main()
