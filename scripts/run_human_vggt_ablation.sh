#!/usr/bin/env bash
set -euo pipefail

SCENE="${1:-human1}"
EXECUTE=0
if [[ "${2:-}" == "--execute" || "${1:-}" == "--execute" ]]; then
  EXECUTE=1
  if [[ "${1:-}" == "--execute" ]]; then
    SCENE="human1"
  fi
fi

IMAGE_DIR="data/${SCENE}/images"
MASK_DIR="data/${SCENE}/masks"
BLACK_SCENE="data/${SCENE}_mask_black"
BLUR_SCENE="data/${SCENE}_mask_blur"
SPARSE_CSV="outputs/tables/human_vggt_ablation.csv"
MASK_CSV="outputs/tables/human_vggt_mask_consistency.csv"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${EXECUTE}" == "1" ]]; then
    "$@"
  fi
}

run_vggt_config() {
  local name="$1"
  local source_scene="$2"
  shift 2
  local output_scene="data/${SCENE}_vggt_${name}"

  run_cmd python scripts/run_vggt_colmap.py \
    --scene_dir "${source_scene}" \
    --output_scene_dir "${output_scene}" \
    --max_images 16 \
    --sample_strategy uniform \
    --use_ba \
    --shared_camera \
    --overwrite_sparse \
    "$@"

  run_cmd python scripts/check_sparse.py "${output_scene}/sparse" \
    --csv "${SPARSE_CSV}" \
    --append_csv

  if [[ -d "${MASK_DIR}" ]]; then
    run_cmd python scripts/eval_sparse_mask_consistency.py \
      --sparse_dir "${output_scene}/sparse" \
      --image_dir "${output_scene}/images" \
      --mask_dir "${MASK_DIR}" \
      --csv "${MASK_CSV}" \
      --append_csv \
      --mask_dilate 3
  fi
}

run_cmd mkdir -p outputs/tables
run_cmd python scripts/make_masked_scene.py \
  --image_dir "${IMAGE_DIR}" \
  --mask_dir "${MASK_DIR}" \
  --output_scene_dir "${BLACK_SCENE}" \
  --mode black
run_cmd python scripts/make_masked_scene.py \
  --image_dir "${IMAGE_DIR}" \
  --mask_dir "${MASK_DIR}" \
  --output_scene_dir "${BLUR_SCENE}" \
  --mode blur

run_vggt_config original_q8 "data/${SCENE}" --query_frame_num 8
run_vggt_config original_q16 "data/${SCENE}" --query_frame_num 16
run_vggt_config mask_black_q16 "${BLACK_SCENE}" --query_frame_num 16
run_vggt_config mask_blur_q16 "${BLUR_SCENE}" --query_frame_num 16
run_vggt_config mask_blur_q16_no_fine "${BLUR_SCENE}" --query_frame_num 16 --no_fine_tracking
run_vggt_config mask_blur_q16_2048pts "${BLUR_SCENE}" --query_frame_num 16 --max_query_pts 2048

if [[ "${EXECUTE}" != "1" ]]; then
  echo "Dry run only. Re-run with: bash scripts/run_human_vggt_ablation.sh ${SCENE} --execute"
fi
