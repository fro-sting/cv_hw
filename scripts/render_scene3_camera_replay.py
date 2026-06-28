#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

CAMERA_MODEL_PARAMS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ImagePose:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render trained 3DGS from the original sampled COLMAP camera poses. "
            "The script writes a nerfstudio camera_path JSON and can optionally call ns-render."
        )
    )
    parser.add_argument(
        "--sparse_dir",
        default="data/scene3_quality_24/sparse_self_ba_intrinsics_pruned",
        help="COLMAP sparse directory used to train splatfacto.",
    )
    parser.add_argument(
        "--dataparser_transforms",
        default="report/artifacts/scene3_self_ba_intrinsics_pruned_24/dataparser_transforms.json",
        help="Nerfstudio dataparser_transforms.json saved beside the trained config.",
    )
    parser.add_argument(
        "--config",
        default="report/artifacts/scene3_self_ba_intrinsics_pruned_24/config.yml",
        help="Trained nerfstudio config.yml. Required only with --render.",
    )
    parser.add_argument(
        "--camera_path",
        default="outputs/scene3_camera_replay/camera_path.json",
        help="Output nerfstudio camera-path JSON.",
    )
    parser.add_argument(
        "--output_video",
        default="outputs/scene3_camera_replay/replay.mp4",
        help="Output video path used by --render.",
    )
    parser.add_argument("--fps", type=float, default=12.0, help="Replay frame rate.")
    parser.add_argument(
        "--downscale",
        type=float,
        default=2.0,
        help="Render resolution divisor relative to COLMAP image size.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Use only the first N poses after sorting by image name; 0 means all poses.",
    )
    parser.add_argument(
        "--no_ns_transform",
        action="store_true",
        help="Do not apply dataparser_transforms.json. Use only for debugging coordinate conventions.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="After writing the camera path, run ns-render camera-path.",
    )
    parser.add_argument(
        "--extra_render_arg",
        action="append",
        default=[],
        help="Extra argument passed to ns-render. Repeat for multiple args, e.g. --extra_render_arg=--rendered-output-names=rgb",
    )
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    return value


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_next_bytes(fid, num_bytes: int, fmt: str):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("Unexpected end of COLMAP binary file.")
    return struct.unpack("<" + fmt, data)


def read_cameras_binary(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    with path.open("rb") as fid:
        (num_cameras,) = read_next_bytes(fid, 8, "Q")
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(fid, 24, "iiQQ")
            if model_id not in CAMERA_MODEL_PARAMS:
                raise ValueError(f"Unsupported COLMAP camera model id {model_id} in {path}")
            model_name, num_params = CAMERA_MODEL_PARAMS[model_id]
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = Camera(camera_id, model_name, width, height, params)
    return cameras


def read_null_terminated_string(fid) -> str:
    chars = bytearray()
    while True:
        char = fid.read(1)
        if not char:
            raise EOFError("Unexpected end of COLMAP image name.")
        if char == b"\x00":
            return chars.decode("utf-8")
        chars.extend(char)


def read_images_binary(path: Path) -> list[ImagePose]:
    images: list[ImagePose] = []
    with path.open("rb") as fid:
        (num_images,) = read_next_bytes(fid, 8, "Q")
        for _ in range(num_images):
            image_id = read_next_bytes(fid, 4, "i")[0]
            qvec = np.array(read_next_bytes(fid, 32, "dddd"), dtype=np.float64)
            tvec = np.array(read_next_bytes(fid, 24, "ddd"), dtype=np.float64)
            camera_id = read_next_bytes(fid, 4, "i")[0]
            name = read_null_terminated_string(fid)
            (num_points2d,) = read_next_bytes(fid, 8, "Q")
            fid.seek(num_points2d * 24, 1)
            images.append(ImagePose(image_id, qvec, tvec, camera_id, name))
    return sorted(images, key=lambda image: image.name)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def colmap_pose_to_nerfstudio_c2w(image: ImagePose) -> np.ndarray:
    rotation = qvec_to_rotmat(image.qvec)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation.T
    c2w[:3, 3] = -rotation.T @ image.tvec
    # COLMAP/OpenCV camera axes are x-right, y-down, z-forward.
    # Nerfstudio camera paths use the OpenGL-style camera convention.
    c2w[:3, 1:3] *= -1.0
    return c2w


def load_dataparser_transform(path: Path) -> tuple[np.ndarray, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :4] = np.asarray(data["transform"], dtype=np.float64)
    scale = float(data.get("scale", 1.0))
    return transform, scale


def apply_ns_dataparser_transform(c2w: np.ndarray, transform: np.ndarray, scale: float) -> np.ndarray:
    out = transform @ c2w
    out[:3, 3] *= scale
    return out


def camera_fov_degrees(camera: Camera) -> float:
    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
        fx = fy = camera.params[0]
    elif camera.model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"}:
        fx, fy = camera.params[:2]
    else:
        raise ValueError(f"Unsupported camera model for FOV conversion: {camera.model}")
    fov_x = 2.0 * math.atan(camera.width / (2.0 * fx))
    fov_y = 2.0 * math.atan(camera.height / (2.0 * fy))
    return math.degrees(max(fov_x, fov_y))


def build_camera_path(
    cameras: dict[int, Camera],
    images: list[ImagePose],
    transform: np.ndarray | None,
    scale: float,
    fps: float,
    downscale: float,
) -> dict:
    if not images:
        raise RuntimeError("No registered COLMAP images found.")

    first_camera = cameras[images[0].camera_id]
    render_width = max(1, int(round(first_camera.width / downscale)))
    render_height = max(1, int(round(first_camera.height / downscale)))
    seconds = max(len(images) / fps, 1.0 / fps)

    path = []
    for image in images:
        camera = cameras[image.camera_id]
        c2w = colmap_pose_to_nerfstudio_c2w(image)
        if transform is not None:
            c2w = apply_ns_dataparser_transform(c2w, transform, scale)
        path.append(
            {
                "camera_to_world": c2w.reshape(-1).tolist(),
                "fov": camera_fov_degrees(camera),
                "aspect": camera.width / camera.height,
                "image_name": image.name,
            }
        )

    return {
        "camera_type": "perspective",
        "render_height": render_height,
        "render_width": render_width,
        "fps": fps,
        "seconds": seconds,
        "smoothness_value": 0.0,
        "is_cycle": False,
        "camera_path": path,
    }


def run_render(config: Path, camera_path: Path, output_video: Path, extra_args: list[str]) -> None:
    if shutil.which("ns-render") is None:
        raise RuntimeError("ns-render not found. Activate the nerfstudio training environment first.")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ns-render",
        "camera-path",
        "--load-config",
        str(config),
        "--camera-path-filename",
        str(camera_path),
        "--output-path",
        str(output_video),
        *extra_args,
    ]
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    sparse_dir = resolve(args.sparse_dir)
    camera_path = resolve(args.camera_path)
    dataparser_transforms = resolve(args.dataparser_transforms)

    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    images = read_images_binary(sparse_dir / "images.bin")
    if args.max_frames > 0:
        images = images[: args.max_frames]

    transform = None
    scale = 1.0
    if not args.no_ns_transform:
        transform, scale = load_dataparser_transform(dataparser_transforms)

    path_json = build_camera_path(cameras, images, transform, scale, args.fps, args.downscale)
    camera_path.parent.mkdir(parents=True, exist_ok=True)
    camera_path.write_text(json.dumps(path_json, indent=2), encoding="utf-8")

    print(f"sparse_dir: {display_path(sparse_dir)}")
    print(f"poses: {len(images)}")
    print(f"resolution: {path_json['render_width']}x{path_json['render_height']}")
    print(f"camera_path: {display_path(camera_path)}")

    if args.render:
        config = resolve(args.config)
        output_video = resolve(args.output_video)
        run_render(config, camera_path, output_video, args.extra_render_arg)
        print(f"output_video: {display_path(output_video)}")


if __name__ == "__main__":
    main()
