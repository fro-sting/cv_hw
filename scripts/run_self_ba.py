#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


@dataclass
class BAProblem:
    image_ids: list[int]
    point3d_ids: list[int]
    camera_params0: np.ndarray
    points0: np.ndarray
    shared_intrinsics0: np.ndarray | None
    shared_camera_model: str | None
    camera_indices: np.ndarray
    point_indices: np.ndarray
    observations: np.ndarray
    intrinsics: list[dict[str, float]]
    fixed_camera_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-implemented Bundle Adjustment for a COLMAP sparse model.")
    parser.add_argument("--input_sparse", required=True, help="Input COLMAP sparse directory.")
    parser.add_argument("--output_sparse", required=True, help="Output sparse directory for optimized reconstruction.")
    parser.add_argument("--metrics_csv", default=None, help="Optional CSV path for before/after metrics.")
    parser.add_argument("--min_track_len", type=int, default=2)
    parser.add_argument("--max_points", type=int, default=3000)
    parser.add_argument("--max_nfev", type=int, default=50)
    parser.add_argument("--loss", default="huber", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"])
    parser.add_argument("--f_scale", type=float, default=2.0)
    parser.add_argument(
        "--optimize_shared_intrinsics",
        action="store_true",
        help="Also optimize one shared SIMPLE_PINHOLE/PINHOLE intrinsic parameter set.",
    )
    parser.add_argument(
        "--prune_reproj_error_px",
        type=float,
        default=None,
        help="After BA, delete optimized 3D points with any observation above this reprojection error.",
    )
    return parser.parse_args()


def camera_intrinsics(camera: pycolmap.Camera) -> dict[str, float]:
    model = camera.model.name if hasattr(camera.model, "name") else str(camera.model)
    params = np.asarray(camera.params, dtype=float)
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        return {"fx": f, "fy": f, "cx": cx, "cy": cy}
    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    raise NotImplementedError(f"Unsupported camera model for this self BA: {model}")


def camera_model_name(camera: pycolmap.Camera) -> str:
    return camera.model.name if hasattr(camera.model, "name") else str(camera.model)


def intrinsics_to_params(intrinsics: dict[str, float], model: str) -> np.ndarray:
    if model == "SIMPLE_PINHOLE":
        return np.asarray([intrinsics["fx"], intrinsics["cx"], intrinsics["cy"]], dtype=float)
    if model == "PINHOLE":
        return np.asarray(
            [intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]],
            dtype=float,
        )
    raise NotImplementedError(f"Unsupported camera model for shared intrinsics: {model}")


def params_to_intrinsics(params: np.ndarray, model: str) -> dict[str, float]:
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        return {"fx": f, "fy": f, "cx": cx, "cy": cy}
    if model == "PINHOLE":
        fx, fy, cx, cy = params
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    raise NotImplementedError(f"Unsupported camera model for shared intrinsics: {model}")


def pose_to_params(image: pycolmap.Image) -> np.ndarray:
    matrix = np.asarray(image.cam_from_world.matrix(), dtype=float)
    rvec = Rotation.from_matrix(matrix[:, :3]).as_rotvec()
    tvec = matrix[:, 3]
    return np.concatenate([rvec, tvec])


def params_to_pose(params: np.ndarray) -> pycolmap.Rigid3d:
    rot = Rotation.from_rotvec(params[:3]).as_matrix()
    tvec = params[3:6]
    return pycolmap.Rigid3d(pycolmap.Rotation3d(rot), tvec)


def project(points: np.ndarray, camera_params: np.ndarray, intrinsics: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    rot = Rotation.from_rotvec(camera_params[:3]).as_matrix()
    tvec = camera_params[3:6]
    cam = points @ rot.T + tvec[None, :]
    z = cam[:, 2]
    valid = z > 1e-8
    pixels = np.empty((len(points), 2), dtype=float)
    pixels[:] = np.nan
    pixels[valid, 0] = intrinsics["fx"] * cam[valid, 0] / z[valid] + intrinsics["cx"]
    pixels[valid, 1] = intrinsics["fy"] * cam[valid, 1] / z[valid] + intrinsics["cy"]
    return pixels, valid


def build_problem(
    reconstruction: pycolmap.Reconstruction,
    min_track_len: int,
    max_points: int,
    optimize_shared_intrinsics: bool,
) -> BAProblem:
    image_ids = sorted(reconstruction.images.keys())
    image_id_to_index = {image_id: idx for idx, image_id in enumerate(image_ids)}
    fixed_camera_index = 0

    camera_params0 = np.stack([pose_to_params(reconstruction.images[image_id]) for image_id in image_ids])
    cameras = [reconstruction.cameras[reconstruction.images[image_id].camera_id] for image_id in image_ids]
    intrinsics = [camera_intrinsics(camera) for camera in cameras]
    shared_intrinsics0 = None
    shared_camera_model = None
    if optimize_shared_intrinsics:
        camera_ids = {camera.camera_id for camera in cameras}
        if len(camera_ids) != 1:
            raise ValueError(
                "Shared intrinsic BA requires all images to share one COLMAP camera. "
                f"Got camera ids: {sorted(camera_ids)}"
            )
        camera_models = {camera_model_name(camera) for camera in cameras}
        if len(camera_models) != 1:
            raise ValueError(f"Shared intrinsic BA requires one camera model, got: {sorted(camera_models)}")
        shared_camera_model = camera_models.pop()
        if shared_camera_model not in {"SIMPLE_PINHOLE", "PINHOLE"}:
            raise NotImplementedError(f"Shared intrinsic BA does not support {shared_camera_model}")
        shared_intrinsics0 = intrinsics_to_params(intrinsics[0], shared_camera_model)

    candidate_ids = [
        point_id
        for point_id, point in reconstruction.points3D.items()
        if len(point.track.elements) >= min_track_len
    ]
    candidate_ids.sort(key=lambda point_id: len(reconstruction.points3D[point_id].track.elements), reverse=True)
    point3d_ids = candidate_ids[:max_points]
    if not point3d_ids:
        raise RuntimeError("No valid 3D points selected for BA.")

    point_id_to_index = {point_id: idx for idx, point_id in enumerate(point3d_ids)}
    points0 = np.stack([reconstruction.points3D[point_id].xyz for point_id in point3d_ids])

    camera_indices: list[int] = []
    point_indices: list[int] = []
    observations: list[np.ndarray] = []
    for point_id in point3d_ids:
        point_idx = point_id_to_index[point_id]
        point = reconstruction.points3D[point_id]
        for element in point.track.elements:
            if element.image_id not in image_id_to_index:
                continue
            image = reconstruction.images[element.image_id]
            point2d = image.points2D[element.point2D_idx]
            if point2d.point3D_id != point_id:
                continue
            camera_indices.append(image_id_to_index[element.image_id])
            point_indices.append(point_idx)
            observations.append(np.asarray(point2d.xy, dtype=float))

    if not observations:
        raise RuntimeError("No observations selected for BA.")

    return BAProblem(
        image_ids=image_ids,
        point3d_ids=point3d_ids,
        camera_params0=camera_params0,
        points0=points0,
        shared_intrinsics0=shared_intrinsics0,
        shared_camera_model=shared_camera_model,
        camera_indices=np.asarray(camera_indices, dtype=np.int32),
        point_indices=np.asarray(point_indices, dtype=np.int32),
        observations=np.asarray(observations, dtype=float),
        intrinsics=intrinsics,
        fixed_camera_index=fixed_camera_index,
    )


def pack(
    problem: BAProblem,
    camera_params: np.ndarray,
    points: np.ndarray,
    shared_intrinsics: np.ndarray | None,
) -> np.ndarray:
    movable = [idx for idx in range(len(problem.image_ids)) if idx != problem.fixed_camera_index]
    pieces = [camera_params[movable].ravel(), points.ravel()]
    if problem.shared_intrinsics0 is not None:
        if shared_intrinsics is None:
            raise ValueError("shared_intrinsics is required for this problem.")
        pieces.append(shared_intrinsics.ravel())
    return np.concatenate(pieces)


def unpack(problem: BAProblem, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]], np.ndarray | None]:
    num_movable = len(problem.image_ids) - 1
    camera_size = num_movable * 6
    point_size = problem.points0.size
    camera_params = problem.camera_params0.copy()
    movable = [idx for idx in range(len(problem.image_ids)) if idx != problem.fixed_camera_index]
    camera_params[movable] = params[:camera_size].reshape((num_movable, 6))
    points = params[camera_size : camera_size + point_size].reshape(problem.points0.shape)
    if problem.shared_intrinsics0 is None:
        return camera_params, points, problem.intrinsics, None

    intrinsics_start = camera_size + point_size
    shared_intrinsics = params[intrinsics_start : intrinsics_start + len(problem.shared_intrinsics0)]
    intrinsics = [params_to_intrinsics(shared_intrinsics, problem.shared_camera_model) for _ in problem.image_ids]
    return camera_params, points, intrinsics, shared_intrinsics


def residuals(params: np.ndarray, problem: BAProblem) -> np.ndarray:
    camera_params, points, intrinsics, _ = unpack(problem, params)
    residual = np.empty((len(problem.observations), 2), dtype=float)
    for camera_idx in range(len(problem.image_ids)):
        obs_mask = problem.camera_indices == camera_idx
        if not np.any(obs_mask):
            continue
        obs_points = points[problem.point_indices[obs_mask]]
        projected, valid = project(obs_points, camera_params[camera_idx], intrinsics[camera_idx])
        diff = projected - problem.observations[obs_mask]
        diff[~valid] = 1e3
        residual[obs_mask] = diff
    return residual.ravel()


def jacobian_sparsity(problem: BAProblem):
    num_obs = len(problem.observations)
    num_movable = len(problem.image_ids) - 1
    num_camera_params = num_movable * 6
    num_point_params = len(problem.point3d_ids) * 3
    num_intrinsics = 0 if problem.shared_intrinsics0 is None else len(problem.shared_intrinsics0)
    sparsity = lil_matrix((num_obs * 2, num_camera_params + num_point_params + num_intrinsics), dtype=int)
    camera_param_offset = {}
    col = 0
    for camera_idx in range(len(problem.image_ids)):
        if camera_idx == problem.fixed_camera_index:
            continue
        camera_param_offset[camera_idx] = col
        col += 6

    for obs_idx, (camera_idx, point_idx) in enumerate(zip(problem.camera_indices, problem.point_indices)):
        rows = [2 * obs_idx, 2 * obs_idx + 1]
        if camera_idx in camera_param_offset:
            c0 = camera_param_offset[camera_idx]
            for r in rows:
                sparsity[r, c0 : c0 + 6] = 1
        p0 = num_camera_params + 3 * point_idx
        for r in rows:
            sparsity[r, p0 : p0 + 3] = 1
        if num_intrinsics:
            i0 = num_camera_params + num_point_params
            for r in rows:
                sparsity[r, i0 : i0 + num_intrinsics] = 1
    return sparsity


def parameter_bounds(problem: BAProblem) -> tuple[np.ndarray, np.ndarray]:
    num_movable = len(problem.image_ids) - 1
    num_camera_params = num_movable * 6
    num_point_params = len(problem.point3d_ids) * 3
    num_intrinsics = 0 if problem.shared_intrinsics0 is None else len(problem.shared_intrinsics0)
    total = num_camera_params + num_point_params + num_intrinsics
    lower = np.full(total, -np.inf, dtype=float)
    upper = np.full(total, np.inf, dtype=float)
    if problem.shared_intrinsics0 is not None:
        offset = num_camera_params + num_point_params
        lower[offset] = 1e-3
        if problem.shared_camera_model == "PINHOLE":
            lower[offset + 1] = 1e-3
    return lower, upper


def error_stats(raw_residuals: np.ndarray) -> dict[str, float]:
    per_obs = np.linalg.norm(raw_residuals.reshape((-1, 2)), axis=1)
    return {
        "mean": float(np.mean(per_obs)),
        "median": float(np.median(per_obs)),
        "p90": float(np.percentile(per_obs, 90)),
    }


def write_metrics(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_points_ply(reconstruction: pycolmap.Reconstruction, path: Path) -> None:
    """Write a minimal ASCII PLY point cloud from reconstruction.points3D."""
    points = list(reconstruction.points3D.values())
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point in points:
            x, y, z = point.xyz
            r, g, b = point.color
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


def update_reconstruction_intrinsics(
    reconstruction: pycolmap.Reconstruction,
    problem: BAProblem,
    shared_intrinsics: np.ndarray | None,
) -> None:
    if shared_intrinsics is None:
        return
    for image_id in problem.image_ids:
        camera = reconstruction.cameras[reconstruction.images[image_id].camera_id]
        model = camera_model_name(camera)
        if model == "SIMPLE_PINHOLE":
            f, cx, cy = shared_intrinsics
            camera.params = np.asarray([f, cx, cy], dtype=float)
        elif model == "PINHOLE":
            camera.params = np.asarray(shared_intrinsics, dtype=float)
        else:
            raise NotImplementedError(f"Unsupported camera model for shared intrinsics: {model}")


def prune_points_by_reprojection_error(
    reconstruction: pycolmap.Reconstruction,
    problem: BAProblem,
    raw_residuals: np.ndarray,
    threshold_px: float,
) -> int:
    per_obs = np.linalg.norm(raw_residuals.reshape((-1, 2)), axis=1)
    bad_point_indices = set(problem.point_indices[per_obs > threshold_px].tolist())
    for point_idx in bad_point_indices:
        reconstruction.delete_point3D(problem.point3d_ids[point_idx])
    return len(bad_point_indices)


def main() -> None:
    args = parse_args()
    started = time.time()
    reconstruction = pycolmap.Reconstruction(args.input_sparse)
    problem = build_problem(
        reconstruction,
        args.min_track_len,
        args.max_points,
        optimize_shared_intrinsics=args.optimize_shared_intrinsics,
    )

    x0 = pack(problem, problem.camera_params0, problem.points0, problem.shared_intrinsics0)
    before = error_stats(residuals(x0, problem))
    print("[Self BA]")
    print(f"input_sparse: {args.input_sparse}")
    print(f"images: {len(problem.image_ids)}")
    print(f"points: {len(problem.point3d_ids)}")
    print(f"observations: {len(problem.observations)}")
    print(f"optimize_shared_intrinsics: {problem.shared_intrinsics0 is not None}")
    if problem.shared_intrinsics0 is not None:
        print(f"shared_camera_model: {problem.shared_camera_model}")
        print(f"shared_intrinsics_before: {problem.shared_intrinsics0.tolist()}")
    print(f"reproj_before_mean: {before['mean']:.6f} px")
    print(f"reproj_before_median: {before['median']:.6f} px")
    print(f"reproj_before_p90: {before['p90']:.6f} px")

    result = least_squares(
        residuals,
        x0,
        args=(problem,),
        jac_sparsity=jacobian_sparsity(problem),
        x_scale="jac",
        bounds=parameter_bounds(problem),
        loss=args.loss,
        f_scale=args.f_scale,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        max_nfev=args.max_nfev,
        verbose=2,
    )
    after_residuals = residuals(result.x, problem)
    after = error_stats(after_residuals)
    camera_params, points, _, shared_intrinsics = unpack(problem, result.x)

    for camera_idx, image_id in enumerate(problem.image_ids):
        reconstruction.images[image_id].cam_from_world = params_to_pose(camera_params[camera_idx])
    for point_idx, point_id in enumerate(problem.point3d_ids):
        reconstruction.points3D[point_id].xyz = points[point_idx]
    update_reconstruction_intrinsics(reconstruction, problem, shared_intrinsics)

    pruned_points = 0
    if args.prune_reproj_error_px is not None:
        pruned_points = prune_points_by_reprojection_error(
            reconstruction,
            problem,
            after_residuals,
            args.prune_reproj_error_px,
        )

    output_sparse = Path(args.output_sparse)
    output_sparse.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(output_sparse))
    write_points_ply(reconstruction, output_sparse / "points.ply")

    elapsed = time.time() - started
    print(f"reproj_after_mean: {after['mean']:.6f} px")
    print(f"reproj_after_median: {after['median']:.6f} px")
    print(f"reproj_after_p90: {after['p90']:.6f} px")
    if shared_intrinsics is not None:
        print(f"shared_intrinsics_after: {shared_intrinsics.tolist()}")
    if args.prune_reproj_error_px is not None:
        print(f"prune_reproj_error_px: {args.prune_reproj_error_px:.6f}")
        print(f"pruned_points: {pruned_points}")
    print(f"success: {result.success}")
    print(f"message: {result.message}")
    print(f"time_sec: {elapsed:.3f}")
    print(f"output_sparse: {output_sparse}")

    if args.metrics_csv:
        write_metrics(
            Path(args.metrics_csv),
            {
                "input_sparse": args.input_sparse,
                "output_sparse": str(output_sparse),
                "num_images": len(problem.image_ids),
                "num_points": len(problem.point3d_ids),
                "num_observations": len(problem.observations),
                "optimize_shared_intrinsics": problem.shared_intrinsics0 is not None,
                "shared_camera_model": problem.shared_camera_model,
                "shared_intrinsics_before": (
                    problem.shared_intrinsics0.tolist() if problem.shared_intrinsics0 is not None else ""
                ),
                "shared_intrinsics_after": shared_intrinsics.tolist() if shared_intrinsics is not None else "",
                "mean_before": before["mean"],
                "median_before": before["median"],
                "p90_before": before["p90"],
                "mean_after": after["mean"],
                "median_after": after["median"],
                "p90_after": after["p90"],
                "prune_reproj_error_px": args.prune_reproj_error_px if args.prune_reproj_error_px is not None else "",
                "pruned_points": pruned_points,
                "time_sec": elapsed,
                "success": result.success,
                "message": result.message,
            },
        )


if __name__ == "__main__":
    main()
