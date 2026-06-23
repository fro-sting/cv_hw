#!/usr/bin/env bash
set -e

cd ~/cv_hw
mkdir -p exports/gs_batch/raw_gs
mkdir -p exports/gs_batch/pointclouds

FOLDERS=(
  "3dgs_human1_ba_masked"
  "3dgs_human1_ba_masked_black"
  "3dgs_human1_raw"
  "3dgs_human1_raw_masked"
  "3dgs_human1_raw_masked_black"
)

for d in "${FOLDERS[@]}"; do
  echo "=============================="
  echo "Processing $d"

  cfg=$(find "outputs/$d" -path "*/splatfacto/*/config.yml" | sort | tail -n 1)

  if [ -z "$cfg" ]; then
    echo "[WARN] No config.yml found for outputs/$d, skip."
    continue
  fi

  echo "[INFO] config: $cfg"

  outdir="exports/gs_batch/raw_gs/$d"
  rm -rf "$outdir"
  mkdir -p "$outdir"

  ns-export gaussian-splat \
    --load-config "$cfg" \
    --output-dir "$outdir"

  ply=$(find "$outdir" -maxdepth 2 -name "*.ply" | head -n 1)

  if [ -z "$ply" ]; then
    echo "[WARN] No exported ply found for $d, skip."
    continue
  fi

  cp "$ply" "exports/gs_batch/raw_gs/${d}_gs.ply"
  echo "[OK] exported raw GS: exports/gs_batch/raw_gs/${d}_gs.ply"
done

echo "=============================="
echo "Raw GS export done."
