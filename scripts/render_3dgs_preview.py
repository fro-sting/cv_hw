#!/usr/bin/env python
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
    parser = argparse.ArgumentParser(description="Render a PNG preview for a 3DGS/Nerfstudio PLY file.")
    parser.add_argument("input", help="Input .ply file.")
    parser.add_argument("output", help="Output .png path.")
    parser.add_argument("--max_points", type=int, default=120000, help="Random point limit for plotting.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--elev", type=float, default=12.0)
    parser.add_argument("--azim", type=float, default=-80.0)
    parser.add_argument("--point_size", type=float, default=0.35)
    parser.add_argument("--opacity_threshold", type=float, default=None, help="Optional sigmoid(opacity) cutoff.")
    parser.add_argument("--white_background", action="store_true")
    return parser.parse_args()


def read_ply_vertices(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Missing PLY end_header in {path}")
            decoded = line.decode("ascii", errors="strict").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        if header_lines[0] != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        if "format binary_little_endian 1.0" not in header_lines:
            raise ValueError("Only binary_little_endian PLY is supported by this preview script.")

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
                    raise ValueError(f"Unsupported PLY property type: {parts[1]}")
                properties.append((parts[2], "<" + dtype))

        if vertex_count is None:
            raise ValueError(f"No vertex element found in {path}")

        data = np.fromfile(f, dtype=np.dtype(properties), count=vertex_count)

    if len(data) != vertex_count:
        raise ValueError(f"Expected {vertex_count} vertices, read {len(data)} from {path}")
    return data


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def colors_from_vertices(vertices: np.ndarray) -> np.ndarray | None:
    names = vertices.dtype.names or ()
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=1).astype(np.float32)
        return np.clip(rgb / 255.0, 0.0, 1.0)
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        sh0 = 0.28209479177387814
        rgb = np.stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]], axis=1).astype(np.float32)
        return np.clip(0.5 + sh0 * rgb, 0.0, 1.0)
    return None


def equalize_axes(ax: plt.Axes, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = max(maxs - mins) * 0.5
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vertices = read_ply_vertices(input_path)
    names = vertices.dtype.names or ()
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY vertex data must contain x, y, z properties.")

    if args.opacity_threshold is not None and "opacity" in names:
        vertices = vertices[sigmoid(vertices["opacity"]) >= args.opacity_threshold]

    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    colors = colors_from_vertices(vertices)
    if len(points) == 0:
        raise RuntimeError(f"No points left to render in {input_path}")

    if args.max_points > 0 and len(points) > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(points), size=args.max_points, replace=False)
        points = points[keep]
        if colors is not None:
            colors = colors[keep]

    fig = plt.figure(figsize=(8, 8), facecolor="white" if args.white_background else "black")
    ax = fig.add_subplot(111, projection="3d", facecolor="white" if args.white_background else "black")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=args.point_size,
        c=colors if colors is not None else points[:, 2],
        cmap=None if colors is not None else "viridis",
        linewidths=0,
        depthshade=False,
    )
    equalize_axes(ax, points)
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout(pad=0)
    fig.savefig(output_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"input: {input_path}")
    print(f"vertices: {len(vertices)}")
    print(f"rendered_points: {len(points)}")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
