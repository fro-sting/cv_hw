#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect report artifacts for one reconstruction experiment.")
    parser.add_argument("--name", required=True, help="Experiment name, e.g. human1_official_ba.")
    parser.add_argument("--sparse_dir", help="Optional COLMAP sparse directory to check and record.")
    parser.add_argument("--metrics_csv", help="Optional BA metrics CSV to copy.")
    parser.add_argument("--data_dir", help="Optional nerfstudio data directory whose README should be copied.")
    parser.add_argument("--config", help="Optional nerfstudio config.yml to copy.")
    parser.add_argument("--export_dir", help="Optional directory containing exported gaussian .ply files.")
    parser.add_argument("--note", default="", help="Optional short note written into summary.md.")
    parser.add_argument("--archive_root", default="report/artifacts", help="Root directory for transferable artifacts.")
    return parser.parse_args()


def resolve(path: str | None) -> Path | None:
    if not path:
        return None
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    return value


def copy_if_exists(src: Path | None, dst: Path) -> None:
    if src is None or not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def run_check_sparse(sparse_dir: Path, output_path: Path) -> None:
    cmd = ["python", str(ROOT / "scripts" / "check_sparse.py"), str(sparse_dir)]
    result = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"check_sparse failed for {sparse_dir}. See {output_path}")


def main() -> None:
    args = parse_args()
    archive_dir = resolve(args.archive_root) / args.name
    archive_dir.mkdir(parents=True, exist_ok=True)

    sparse_dir = resolve(args.sparse_dir)
    metrics_csv = resolve(args.metrics_csv)
    data_dir = resolve(args.data_dir)
    config = resolve(args.config)
    export_dir = resolve(args.export_dir)

    if sparse_dir is not None:
        run_check_sparse(sparse_dir, archive_dir / "check_sparse.txt")
    copy_if_exists(metrics_csv, archive_dir / "metrics.csv")
    if data_dir is not None:
        copy_if_exists(data_dir / "README.md", archive_dir / "nerfstudio_data_README.md")
    copy_if_exists(config, archive_dir / "config.yml")

    if export_dir is not None and export_dir.exists():
        for ply in sorted(export_dir.rglob("*.ply")):
            copy_if_exists(ply, archive_dir / ply.name)

    summary = [
        f"# {args.name}",
        "",
        f"sparse_dir: `{args.sparse_dir or ''}`",
        f"metrics_csv: `{args.metrics_csv or ''}`",
        f"data_dir: `{args.data_dir or ''}`",
        f"config: `{args.config or ''}`",
        f"export_dir: `{args.export_dir or ''}`",
    ]
    if args.note:
        summary.extend(["", args.note])
    (archive_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"archive_dir: {archive_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
