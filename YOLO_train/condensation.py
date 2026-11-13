"""EXPERIMENTAL: YOLO detections + Condensation (particle filter) tracker.

Each track keeps N particles with state [cx, cy, dx, dy, w, h]; particles are
perturbed each frame, re-weighted by IoU with the matched detection, and
resampled. Tracks are associated to detections with the Hungarian algorithm on
1 - IoU. Kept as an experiment; track_sort.py is the supported tracker.

    python condensation.py --video ../dataset/test_videos/test001.mp4
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "YOLO_train" / "runs" / "train" / "hexbug_train10" / "weights" / "best.pt"


# ------------------- Tracker Classes ------------------- #

class ParticleTrack:
    def __init__(self, id, bbox, n_particles=100):
        self.id = id
        self.n_particles = n_particles
        self.particles = [self.init_state_from_bbox(bbox) for _ in range(n_particles)]
        self.weights = np.ones(n_particles) / n_particles
        self.bbox = bbox
        self.age = 0
        self.time_since_update = 0

    def init_state_from_bbox(self, bbox):
        x, y, w, h = bbox
        return np.array([x + w / 2, y + h / 2, 0, 0, w, h])

    def predict(self):
        for i in range(self.n_particles):
            x, y, dx, dy, w, h = self.particles[i]
            x += dx + np.random.normal(0, 2)
            y += dy + np.random.normal(0, 2)
            dx += np.random.normal(0, 1)
            dy += np.random.normal(0, 1)
            self.particles[i] = np.array([x, y, dx, dy, w, h])

    def update(self, detections):
        weights = []
        for p in self.particles:
            px, py, _, _, w, h = p
            pred_bbox = [px - w / 2, py - h / 2, w, h]
            ious = [compute_iou(pred_bbox, det) for det in detections]
            weights.append(max(ious) if ious else 0.0)
        weights = np.clip(weights, 1e-6, 1.0)
        self.weights = np.array(weights)
        self.weights /= np.sum(self.weights)
        self.resample()
        self.estimate_state()

    def resample(self):
        idxs = np.random.choice(range(self.n_particles), size=self.n_particles, p=self.weights)
        self.particles = [self.particles[i] for i in idxs]
        self.weights = np.ones(self.n_particles) / self.n_particles

    def estimate_state(self):
        state = np.average(self.particles, axis=0, weights=self.weights)
        cx, cy, _, _, w, h = state
        self.bbox = [int(cx - w / 2), int(cy - h / 2), int(w), int(h)]


# ------------------- Utility Functions ------------------- #

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xb - xa) * max(0, yb - ya)
    return inter_area / float(w1 * h1 + w2 * h2 - inter_area + 1e-6)


def draw_tracks(frame, tracks):
    for t in tracks:
        x, y, w, h = t.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {t.id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame


# ------------------- Main Loop ------------------- #

def run(weights, video_path, output_path, conf, device):
    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video_path))
    tracks = []
    next_id = 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps if fps > 0 else 30.0, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=conf, iou=0.45, device=device, verbose=False)[0]
        detections = []
        for x, y, w, h in results.boxes.xywh.cpu().numpy():
            detections.append([int(x - w / 2), int(y - h / 2), int(w), int(h)])

        for track in tracks:
            track.predict()

        cost_matrix = [[1 - compute_iou(track.bbox, det) for det in detections] for track in tracks]
        matched_tracks, matched_dets = set(), set()

        if cost_matrix and detections:
            cost_matrix = np.array(cost_matrix)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 0.5:
                    tracks[r].update([detections[c]])
                    tracks[r].time_since_update = 0
                    matched_tracks.add(r)
                    matched_dets.add(c)

        for i, track in enumerate(tracks):
            if i not in matched_tracks:
                track.time_since_update += 1

        tracks = [t for t in tracks if t.time_since_update <= 10]

        for j, det in enumerate(detections):
            if j not in matched_dets:
                tracks.append(ParticleTrack(next_id, det))
                next_id += 1

        out.write(draw_tracks(frame, tracks))

    cap.release()
    out.release()
    print(f"[INFO] Tracking video saved to: {output_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, type=Path)
    ap.add_argument("--video", default=ROOT / "dataset" / "test_videos" / "test001.mp4", type=Path)
    ap.add_argument("--out", default=None, type=Path, help="default: YOLO_train/ran_tests/<video>_particle.mp4")
    ap.add_argument("--conf", default=0.2, type=float)
    ap.add_argument("--device", default=0 if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_path = args.out or ROOT / "YOLO_train" / "ran_tests" / f"{args.video.stem}_particle.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run(args.weights, args.video, out_path, args.conf, args.device)


if __name__ == "__main__":
    main()
