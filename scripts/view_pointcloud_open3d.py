#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


PLY_NUMPY_DTYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "i2",
    "ushort": "u2",
    "int16": "i2",
    "uint16": "u2",
    "int": "i4",
    "uint": "u4",
    "int32": "i4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a point cloud with Open3D.")
    parser.add_argument("input", help="Input point cloud, usually sparse/points.ply.")
    parser.add_argument("--voxel_size", type=float, default=0.0, help="Optional voxel downsample size.")
    parser.add_argument("--max_points", type=int, default=0, help="Optional random point limit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--point_size", type=float, default=2.0)
    parser.add_argument("--background", choices=["black", "white"], default="white")
    parser.add_argument("--saturation", type=float, default=1.0, help="Display-only color saturation multiplier.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Display-only gamma correction. Values below 1 brighten colors.")
    parser.add_argument("--screenshot", default=None, help="Optional screenshot path.")
    return parser.parse_args()


def read_binary_little_endian_vertices(path: Path) -> np.ndarray | None:
    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                return None
            decoded = line.decode("ascii", errors="strict").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        if header_lines[0] != "ply" or "format binary_little_endian 1.0" not in header_lines:
            return None

        vertex_count = None
        properties: list[tuple[str, str]] = []
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                continue
            if in_vertex and len(parts) == 3 and parts[0] == "property":
                dtype = PLY_NUMPY_DTYPES.get(parts[1])
                if dtype is None:
                    return None
                properties.append((parts[2], "<" + dtype))

        if vertex_count is None:
            return None
        return np.fromfile(f, dtype=np.dtype(properties), count=vertex_count)


def colors_from_3dgs(path: Path, point_count: int) -> np.ndarray | None:
    vertices = read_binary_little_endian_vertices(path)
    if vertices is None or len(vertices) != point_count:
        return None
    names = vertices.dtype.names or ()
    if not {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        return None

    sh0 = 0.28209479177387814
    colors = np.stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]], axis=1).astype(np.float32)
    return np.clip(0.5 + sh0 * colors, 0.0, 1.0)


def adjust_display_colors(colors: np.ndarray, saturation: float, gamma: float) -> np.ndarray:
    colors = np.clip(colors, 0.0, 1.0)
    if saturation != 1.0:
        gray = colors.mean(axis=1, keepdims=True)
        colors = gray + saturation * (colors - gray)
    if gamma != 1.0:
        if gamma <= 0:
            raise ValueError("--gamma must be greater than 0")
        colors = np.power(np.clip(colors, 0.0, 1.0), gamma)
    return np.clip(colors, 0.0, 1.0)


def limit_points(pcd: o3d.geometry.PointCloud, max_points: int, seed: int) -> o3d.geometry.PointCloud:
    if max_points <= 0 or len(pcd.points) <= max_points:
        return pcd
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(pcd.points), size=max_points, replace=False)
    return pcd.select_by_index(keep.tolist())


def load_point_cloud(
    path: Path,
    voxel_size: float,
    max_points: int,
    seed: int,
    saturation: float,
    gamma: float,
) -> o3d.geometry.PointCloud:
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise RuntimeError(f"Open3D loaded an empty point cloud: {path}")

    colors = colors_from_3dgs(path, len(pcd.points))
    if colors is not None:
        colors = adjust_display_colors(colors, saturation=saturation, gamma=gamma)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print("Loaded 3DGS colors from f_dc_0/f_dc_1/f_dc_2.")
        print(
            "RGB mean:",
            np.asarray(pcd.colors).mean(axis=0),
            "luminance mean:",
            np.asarray(pcd.colors).mean(axis=1).mean(),
        )
    elif not pcd.has_colors():
        pcd.paint_uniform_color([0.85, 0.85, 0.85])
        print("No colors found; painted points light gray.")

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd = limit_points(pcd, max_points=max_points, seed=seed)
    return pcd


def main() -> None:
    args = parse_args()
    pcd = load_point_cloud(
        Path(args.input),
        args.voxel_size,
        args.max_points,
        args.seed,
        args.saturation,
        args.gamma,
    )

    print(f"points: {len(pcd.points)}")
    print("Controls: left-drag rotate, wheel zoom, right-drag/pan translate, Q/Esc close.")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Open3D Point Cloud", width=1280, height=900)
    vis.add_geometry(pcd)

    render_options = vis.get_render_option()
    render_options.point_size = args.point_size
    render_options.point_color_option = o3d.visualization.PointColorOption.Color
    if args.background == "white":
        render_options.background_color = np.asarray([1.0, 1.0, 1.0])
    else:
        render_options.background_color = np.asarray([0.0, 0.0, 0.0])

    vis.run()
    if args.screenshot:
        out = Path(args.screenshot)
        out.parent.mkdir(parents=True, exist_ok=True)
        vis.capture_screen_image(str(out), do_render=True)
        print(f"wrote: {out}")
    vis.destroy_window()


if __name__ == "__main__":
    main()
