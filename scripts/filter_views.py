#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import cv2
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality-aware keyframe selection for indoor VGGT scenes.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image_dir", help="Directory containing candidate frames.")
    source.add_argument("--video", help="Video to sample before scoring.")
    parser.add_argument("--output_dir", required=True, help="Output scene directory. Writes images/ inside it.")
    parser.add_argument("--csv", help="Per-frame score CSV. Defaults to output_dir/view_scores.csv.")
    parser.add_argument("--figure", help="Score figure path. Defaults to output_dir/view_scores.png.")
    parser.add_argument("--max_frames", type=int, default=24, help="Maximum selected frames.")
    parser.add_argument(
        "--strategy",
        choices=["quality", "quality_segmented", "uniform", "fixed_interval"],
        default="quality",
    )
    parser.add_argument("--fixed_interval", type=int, default=10, help="Frame step for fixed_interval strategy.")
    parser.add_argument("--fps", type=float, default=2.0, help="Sampling FPS when --video is used.")
    parser.add_argument("--resize_long_edge", type=int, default=1600, help="Resize selected output images.")
    parser.add_argument("--hist_bins", type=int, default=32)
    parser.add_argument("--min_hist_diff", type=float, default=0.08, help="Minimum histogram distance between selected neighbors.")
    parser.add_argument(
        "--min_gap_frames",
        type=int,
        default=0,
        help="Minimum frame-index gap for quality_segmented selections. Use 0 to disable.",
    )
    parser.add_argument("--quality_weight", type=float, default=1.0)
    parser.add_argument("--diff_weight", type=float, default=0.35)
    return parser.parse_args()


def collect_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def sample_video(video: Path, fps: float) -> list[tuple[str, np.ndarray]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    step = 1 if source_fps <= 0 or fps <= 0 else max(1, round(source_fps / fps))
    frames: list[tuple[str, np.ndarray]] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            frames.append((f"{frame_idx:06d}.jpg", frame))
        frame_idx += 1
    cap.release()
    return frames


def load_frames(args: argparse.Namespace) -> list[tuple[str, np.ndarray]]:
    if args.video:
        return sample_video(Path(args.video), args.fps)
    frames = []
    for path in collect_images(Path(args.image_dir)):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            frames.append((path.name, image))
    return frames


def sharpness_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def color_hist(image: np.ndarray, bins: int) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def uniform_indices(total: int, count: int) -> list[int]:
    if count >= total:
        return list(range(total))
    if count == 1:
        return [0]
    return sorted({round(i * (total - 1) / (count - 1)) for i in range(count)})


def quality_segmented_indices(
    total: int,
    count: int,
    scores: list[dict[str, float]],
    min_gap_frames: int,
) -> list[int]:
    if count >= total:
        return list(range(total))
    selected: list[int] = []
    min_gap_frames = max(0, min_gap_frames)
    for segment_idx in range(count):
        start = round(segment_idx * total / count)
        end = round((segment_idx + 1) * total / count)
        if end <= start:
            end = min(total, start + 1)
        segment = sorted(range(start, end), key=lambda idx: scores[idx]["combined_score"], reverse=True)
        chosen = segment[0]
        if min_gap_frames > 0:
            for idx in segment:
                if all(abs(idx - other) >= min_gap_frames for other in selected):
                    chosen = idx
                    break
        selected.append(chosen)
    return sorted(set(selected))


def select_indices(args: argparse.Namespace, scores: list[dict[str, float]]) -> list[int]:
    total = len(scores)
    count = min(args.max_frames, total)
    if args.strategy == "uniform":
        return uniform_indices(total, count)
    if args.strategy == "fixed_interval":
        indices = list(range(0, total, max(1, args.fixed_interval)))
        return indices[:count]
    if args.strategy == "quality_segmented":
        return quality_segmented_indices(total, count, scores, args.min_gap_frames)

    ranked = sorted(range(total), key=lambda idx: scores[idx]["combined_score"], reverse=True)
    selected: list[int] = []
    for idx in ranked:
        if all(abs(idx - other) > 1 or scores[max(idx, other)]["prev_hist_diff"] >= args.min_hist_diff for other in selected):
            selected.append(idx)
        if len(selected) >= count:
            break
    if len(selected) < count:
        for idx in ranked:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= count:
                break
    return sorted(selected)


def resize_image(image: np.ndarray, long_edge: int) -> np.ndarray:
    if long_edge <= 0:
        return image
    h, w = image.shape[:2]
    scale = long_edge / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def write_scores(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "filename",
        "sharpness",
        "sharpness_norm",
        "prev_hist_diff",
        "prev_hist_diff_norm",
        "combined_score",
        "selected",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_figure(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = [int(row["index"]) for row in rows]
    combined = [float(row["combined_score"]) for row in rows]
    sharp = [float(row["sharpness_norm"]) for row in rows]
    diff = [float(row["prev_hist_diff_norm"]) for row in rows]
    selected_x = [int(row["index"]) for row in rows if row["selected"]]
    selected_y = [float(row["combined_score"]) for row in rows if row["selected"]]

    plt.figure(figsize=(10, 4))
    plt.plot(indices, combined, label="combined")
    plt.plot(indices, sharp, label="sharpness", alpha=0.75)
    plt.plot(indices, diff, label="hist diff", alpha=0.75)
    plt.scatter(selected_x, selected_y, s=28, label="selected", zorder=3)
    plt.xlabel("frame index")
    plt.ylabel("normalized score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    frames = load_frames(args)
    if not frames:
        raise RuntimeError("No candidate frames found.")

    sharpness = [sharpness_score(image) for _, image in frames]
    hists = [color_hist(image, args.hist_bins) for _, image in frames]
    prev_diffs = [0.0]
    prev_diffs.extend(hist_distance(hists[idx - 1], hists[idx]) for idx in range(1, len(hists)))
    sharpness_norm = normalize(sharpness)
    diff_norm = normalize(prev_diffs)

    scores: list[dict[str, float]] = []
    for idx in range(len(frames)):
        combined = args.quality_weight * sharpness_norm[idx] + args.diff_weight * diff_norm[idx]
        scores.append(
            {
                "sharpness": sharpness[idx],
                "sharpness_norm": sharpness_norm[idx],
                "prev_hist_diff": prev_diffs[idx],
                "prev_hist_diff_norm": diff_norm[idx],
                "combined_score": combined,
            }
        )

    selected = set(select_indices(args, scores))
    output_image_dir = Path(args.output_dir) / "images"
    if output_image_dir.exists():
        shutil.rmtree(output_image_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for idx, (filename, image) in enumerate(frames):
        is_selected = idx in selected
        if is_selected:
            output = resize_image(image, args.resize_long_edge)
            out_name = f"{len([row for row in rows if row['selected']]) + 1:06d}{Path(filename).suffix or '.jpg'}"
            cv2.imwrite(str(output_image_dir / out_name), output)
        rows.append(
            {
                "index": idx,
                "filename": filename,
                "sharpness": f"{scores[idx]['sharpness']:.6f}",
                "sharpness_norm": f"{scores[idx]['sharpness_norm']:.6f}",
                "prev_hist_diff": f"{scores[idx]['prev_hist_diff']:.6f}",
                "prev_hist_diff_norm": f"{scores[idx]['prev_hist_diff_norm']:.6f}",
                "combined_score": f"{scores[idx]['combined_score']:.6f}",
                "selected": is_selected,
            }
        )

    csv_path = Path(args.csv) if args.csv else Path(args.output_dir) / "view_scores.csv"
    figure_path = Path(args.figure) if args.figure else Path(args.output_dir) / "view_scores.png"
    write_scores(csv_path, rows)
    write_figure(figure_path, rows)

    print("View filtering summary")
    print(f"  candidates: {len(frames)}")
    print(f"  selected: {len(selected)}")
    print(f"  strategy: {args.strategy}")
    print(f"  output_images: {output_image_dir}")
    print(f"  csv: {csv_path}")
    print(f"  figure: {figure_path}")


if __name__ == "__main__":
    main()
