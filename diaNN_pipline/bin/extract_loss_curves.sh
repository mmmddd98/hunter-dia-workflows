#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash bin/extract_loss_curves.sh --log <run.log> --outdir <dir> --lr-tag <tag> [--diann-version <ver>]

Description:
  Parses DIA-NN tuning log sections for IM/RT/FR and extracts lines like:
    1%: loss = 0.0255027
  into per-model CSV files and simple PNG plots.

  If multiple distillation rounds are present (e.g. d0/d1/d2), rounds are
  segmented and stored/plotted separately.

Outputs (for each detected model in log):
  <outdir>/LR_<lr-tag>_<MODEL>_LOSS.csv
  <outdir>/LR_<lr-tag>_<MODEL>_LOSS.png

Examples:
  bash bin/extract_loss_curves.sh \
    --log /path/to/run.log \
    --outdir /path/to/lr_0p001 \
    --lr-tag 0p001
EOF
}

LOG_FILE=""
OUTDIR=""
LR_TAG=""
DIANN_VERSION=""
PLOT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log)
      shift
      LOG_FILE="${1:-}"
      ;;
    --outdir)
      shift
      OUTDIR="${1:-}"
      ;;
    --lr-tag)
      shift
      LR_TAG="${1:-}"
      ;;
    --diann-version)
      shift
      DIANN_VERSION="${1:-}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

[[ -n "$LOG_FILE" ]] || { echo "ERROR: --log is required" >&2; exit 1; }
[[ -n "$OUTDIR" ]] || { echo "ERROR: --outdir is required" >&2; exit 1; }
[[ -n "$LR_TAG" ]] || { echo "ERROR: --lr-tag is required" >&2; exit 1; }
[[ -f "$LOG_FILE" ]] || { echo "ERROR: Missing log file: $LOG_FILE" >&2; exit 1; }
mkdir -p "$OUTDIR"
PLOT_DIR="${LOSS_PLOTS_DIR:-${OUTDIR}/loss_plots}"
mkdir -p "$PLOT_DIR"

if [[ -n "$DIANN_VERSION" ]]; then
  DIANN_VERSION="${DIANN_VERSION#diann-}"
fi

TMP_COMBINED="${OUTDIR}/.loss_combined_${LR_TAG}.tmp"

awk -v diann_version="$DIANN_VERSION" '
  {
    line = tolower($0)
    if (line ~ /fine-tuning the im model/ || line ~ /fine-tuning the ion mobility model/) {
      model = "IM"
      next
    }
    if (line ~ /fine-tuning the rt model/ || line ~ /fine-tuning the retention time model/) {
      model = "RT"
      next
    }
    if (line ~ /fine-tuning the fr model/ || line ~ /fine-tuning the fragmentation model/) {
      model = "FR"
      next
    }
  }
  {
    if (model == "") next
    if (match($0, /^([0-9]+)%:[[:space:]]*loss[[:space:]]*=[[:space:]]*([-+0-9.eE]+)/, m)) {
      percent = m[1] + 0
      if (!(model in seen_model)) {
        seen_model[model] = 1
        distill[model] = 0
      } else if (diann_version >= "2.5.0" && percent < last_percent[model]) {
        distill[model] += 1
      }
      last_percent[model] = percent
      print model "," distill[model] "," m[1] "," m[2]
    }
  }
' "$LOG_FILE" > "$TMP_COMBINED"

if [[ ! -s "$TMP_COMBINED" ]]; then
  rm -f "$TMP_COMBINED"
  echo "ERROR: No loss entries detected in $LOG_FILE" >&2
  exit 1
fi

created_any=0
for model in IM RT FR; do
  count="$(awk -F',' -v m="$model" '$1 == m { c += 1 } END { print c + 0 }' "$TMP_COMBINED")"
  if [[ "$count" -eq 0 ]]; then
    continue
  fi

  created_any=1
  csv_file="${OUTDIR}/LR_${LR_TAG}_${model}_LOSS.csv"
  png_file="${PLOT_DIR}/LR_${LR_TAG}_${model}_LOSS.png"

  {
    echo "distill,percent,loss"
    awk -F',' -v m="$model" '$1 == m { print $2 "," $3 "," $4 }' "$TMP_COMBINED"
  } > "$csv_file"

    python3 - "$csv_file" "$png_file" "$model" "$LR_TAG" "$DIANN_VERSION" <<'PY'
import csv
import sys
from collections import defaultdict

csv_file, png_file, model, lr_tag, diann_version = sys.argv[1:6]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit(f"Matplotlib is required for plotting ({exc})")

curves = defaultdict(lambda: ([], []))
with open(csv_file, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        distill = int(row.get("distill", "0"))
        x, y = curves[distill]
        x.append(float(row["percent"]))
        y.append(float(row["loss"]))

if not curves:
    raise SystemExit(f"No data rows in {csv_file}")

plt.figure(figsize=(7, 4))
for distill in sorted(curves):
    x, y = curves[distill]
    plt.plot(x, y, marker="o", linewidth=1.2, markersize=2.8, label=f"d{distill}")
plt.xlabel("Training progress [%]")
plt.ylabel("Loss")
plt.title(f"LR {lr_tag} - {model} loss")
plt.grid(True, alpha=0.35)
if len(curves) > 1:
    plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(png_file, dpi=140)
PY
done

rm -f "$TMP_COMBINED"

if [[ "$created_any" -ne 1 ]]; then
  echo "ERROR: No IM/RT/FR loss blocks found in $LOG_FILE" >&2
  exit 1
fi

echo "Created loss CSV/plots in: $OUTDIR"
