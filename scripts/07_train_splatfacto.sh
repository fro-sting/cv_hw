#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=""
OUTPUT_DIR="outputs/3dgs"
EXPERIMENT_NAME=""
MAX_NUM_ITERATIONS=""
COLMAP_PATH="sparse/0"
DOWNSCALE_FACTOR="2"
MASKS_PATH="auto"
BACKGROUND_COLOR="black"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)
      DATA_DIR="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --experiment-name)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --max-num-iterations)
      MAX_NUM_ITERATIONS="$2"
      shift 2
      ;;
    --colmap-path)
      COLMAP_PATH="$2"
      shift 2
      ;;
    --downscale-factor)
      DOWNSCALE_FACTOR="$2"
      shift 2
      ;;
    --masks-path)
      MASKS_PATH="$2"
      shift 2
      ;;
    --background-color)
      BACKGROUND_COLOR="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$DATA_DIR" ]]; then
  echo "Usage: bash scripts/07_train_splatfacto.sh --data DATA_DIR [--output outputs/3dgs] [--experiment-name NAME] [--max-num-iterations N] [-- extra ns-train args]" >&2
  exit 2
fi

if ! command -v ns-train >/dev/null 2>&1; then
  echo "ns-train not found. Install nerfstudio in this environment first, e.g. pip install nerfstudio" >&2
  exit 127
fi

python -c "from gsplat.cuda._backend import _C; assert _C is not None and hasattr(_C, 'CameraModelType'), 'gsplat CUDA extension is unavailable. Install CUDA toolkit and reinstall a CUDA-enabled gsplat build before running splatfacto.'" >/dev/null

ns-train splatfacto --help >/dev/null
ns-train splatfacto colmap --help >/dev/null

CMD=(ns-train splatfacto --output-dir "$OUTPUT_DIR")
if [[ -n "$EXPERIMENT_NAME" ]]; then
  CMD+=(--experiment-name "$EXPERIMENT_NAME")
fi
if [[ -n "$MAX_NUM_ITERATIONS" ]]; then
  CMD+=(--max-num-iterations "$MAX_NUM_ITERATIONS")
fi
if [[ -n "$BACKGROUND_COLOR" ]]; then
  CMD+=(--pipeline.model.background-color "$BACKGROUND_COLOR")
fi
CMD+=("${EXTRA_ARGS[@]}")
CMD+=(colmap --data "$DATA_DIR" --colmap-path "$COLMAP_PATH")
if [[ -n "$DOWNSCALE_FACTOR" ]]; then
  CMD+=(--downscale-factor "$DOWNSCALE_FACTOR")
fi
if [[ "$MASKS_PATH" == "auto" ]]; then
  if [[ -d "$DATA_DIR/masks" ]]; then
    CMD+=(--masks-path masks)
  fi
elif [[ -n "$MASKS_PATH" && "$MASKS_PATH" != "none" ]]; then
  CMD+=(--masks-path "$MASKS_PATH")
fi

echo "Running:"
printf ' %q' "${CMD[@]}"
echo
exec "${CMD[@]}"
