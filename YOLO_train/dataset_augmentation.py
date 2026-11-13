"""Offline augmentation: write one augmented copy of every training image + label.

Run AFTER split_dataset.py so validation frames are never augmented. Output
goes next to the originals as <name>_aug<N>.jpg / .txt.

    python dataset_augmentation.py
"""
import argparse
from pathlib import Path

import albumentations as A
import cv2

ROOT = Path(__file__).resolve().parents[1]

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.5),
    A.Blur(blur_limit=3, p=0.2),
    A.Affine(translate_percent=0.05, scale=(0.95, 1.05), rotate=(-10, 10), p=0.5),
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))


def read_yolo_label(label_path: Path):
    bboxes, class_labels = [], []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id, xc, yc, w, h = map(float, parts)
        bboxes.append([xc, yc, w, h])
        class_labels.append(int(class_id))
    return bboxes, class_labels


def clip_bbox(bbox):
    return [min(max(v, 0.0), 1.0) for v in bbox]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=ROOT / "YOLO_train" / "images" / "train", type=Path)
    ap.add_argument("--labels", default=ROOT / "YOLO_train" / "labels" / "train", type=Path)
    args = ap.parse_args()

    image_paths = [p for p in args.images.iterdir() if p.suffix.lower() in (".png", ".jpg") and "_aug" not in p.stem]
    print(f"Found {len(image_paths)} training images.")

    aug_count = 0
    for image_path in image_paths:
        label_path = args.labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue  # unlabelled (background) frame; nothing to augment

        bboxes, class_labels = read_yolo_label(label_path)
        if not bboxes:
            continue

        image = cv2.imread(str(image_path))
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_bboxes = [clip_bbox(b) for b in augmented["bboxes"]]
        if not aug_bboxes:
            continue

        stem = f"{image_path.stem}_aug{aug_count}"
        cv2.imwrite(str(args.images / f"{stem}.jpg"), augmented["image"])
        with open(args.labels / f"{stem}.txt", "w") as f:
            for label, bbox in zip(augmented["class_labels"], aug_bboxes):
                f.write(f"{label} {' '.join(f'{x:.6f}' for x in bbox)}\n")
        aug_count += 1

    print(f"[✓] Augmentation complete. {aug_count} augmented samples saved.")


if __name__ == "__main__":
    main()
