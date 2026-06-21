#!/usr/bin/env bash
set -euo pipefail

SEARCH_DIR=""
CONFIG=""
OUTPUT_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --search-dir)
      SEARCH_DIR="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
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

if [[ -z "$CONFIG" && -z "$SEARCH_DIR" ]]; then
  echo "Usage: bash scripts/08_export_splat.sh (--config CONFIG.yml | --search-dir outputs/3dgs/EXP) --output OUTPUT_DIR [-- extra ns-export args]" >&2
  exit 2
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "--output is required" >&2
  exit 2
fi

if ! command -v ns-export >/dev/null 2>&1; then
  echo "ns-export not found. Install nerfstudio in this environment first, e.g. pip install nerfstudio" >&2
  exit 127
fi

if [[ -z "$CONFIG" ]]; then
  CONFIG="$(find "$SEARCH_DIR" -name config.yml -type f -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi

if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
  echo "Could not find config.yml. search-dir=$SEARCH_DIR config=$CONFIG" >&2
  exit 1
fi

ns-export gaussian-splat --help >/dev/null

CMD=(ns-export gaussian-splat --load-config "$CONFIG" --output-dir "$OUTPUT_DIR")
CMD+=("${EXTRA_ARGS[@]}")

echo "Running:"
printf ' %q' "${CMD[@]}"
echo
exec "${CMD[@]}"
