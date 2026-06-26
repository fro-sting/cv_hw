#!/usr/bin/env bash
set -euo pipefail

SCENE="${1:-scene3}"
EXECUTE=0
if [[ "${2:-}" == "--execute" || "${1:-}" == "--execute" ]]; then
  EXECUTE=1
  if [[ "${1:-}" == "--execute" ]]; then
    SCENE="scene3"
  fi
fi

BASE_DIR="data/${SCENE}"
VIDEO="${BASE_DIR}/source_video.mp4"
IMAGE_DIR="${BASE_DIR}/images"
SPARSE_CSV="outputs/tables/scene3_vggt_ablation.csv"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${EXECUTE}" == "1" ]]; then
    "$@"
  fi
}

source_args() {
  if [[ -f "${VIDEO}" ]]; then
    printf '%s\n' "--video" "${VIDEO}"
  else
    printf '%s\n' "--image_dir" "${IMAGE_DIR}"
  fi
}

prepare_views() {
  local name="$1"
  local max_frames="$2"
  local strategy="$3"
  local fixed_interval="${4:-10}"
  local fps="${5:-2}"
  local output_scene="data/${SCENE}_${name}"
  mapfile -t source < <(source_args)

  run_cmd python scripts/10_filter_views.py \
    "${source[@]}" \
    --output_dir "${output_scene}" \
    --max_frames "${max_frames}" \
    --strategy "${strategy}" \
    --fixed_interval "${fixed_interval}" \
    --fps "${fps}" \
    --csv "${output_scene}/view_scores.csv" \
    --figure "${output_scene}/view_scores.png"
}

run_vggt() {
  local name="$1"
  local query_frames="${2:-5}"
  local self_ba_max_points="${3:-12000}"
  local scene_dir="data/${SCENE}_${name}"
  local self_ba_name="sparse_self_ba_intrinsics_pruned"

  run_cmd python scripts/run_vggt_colmap.py \
    --scene_dir "${scene_dir}" \
    --use_ba \
    --skip_builtin_ba \
    --shared_camera \
    --query_frame_num "${query_frames}" \
    --max_query_pts 2048 \
    --no_fine_tracking \
    --overwrite_sparse \
    --run_self_ba \
    --self_ba_output_name "${self_ba_name}" \
    --self_ba_max_points "${self_ba_max_points}" \
    --self_ba_max_nfev 80 \
    --self_ba_loss soft_l1 \
    --self_ba_f_scale 2.0 \
    --self_ba_max_initial_reproj_error_px 64 \
    --self_ba_prune_reproj_error_px 8 \
    --self_ba_optimize_shared_intrinsics \
    --overwrite_self_ba_output

  run_cmd python scripts/check_sparse.py "${scene_dir}/${self_ba_name}" \
    --csv "${SPARSE_CSV}" \
    --append_csv
}

run_cmd mkdir -p outputs/tables

prepare_views fixed_interval_24 24 fixed_interval 2
prepare_views uniform_24 24 uniform
prepare_views quality_24 24 quality
prepare_views quality_48 48 quality
prepare_views quality_96 96 quality 10 3

run_vggt fixed_interval_24 5 12000
run_vggt uniform_24 5 12000
run_vggt quality_24 5 12000
run_vggt quality_48 5 12000
run_vggt quality_96 8 16000

if [[ "${EXECUTE}" != "1" ]]; then
  echo "Dry run only. Re-run with: bash scripts/run_scene3_vggt_ablation.sh ${SCENE} --execute"
fi
