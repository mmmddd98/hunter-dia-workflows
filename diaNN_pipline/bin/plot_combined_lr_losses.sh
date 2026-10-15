#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash bin/plot_combined_lr_losses.sh --s3dir <OUTBASE/<expname>/03_tune_models>

Description:
  Scans all lr_* subfolders for files named:
    LR_<lr-tag>_<MODEL>_LOSS.csv
  and creates combined plots split by distillation round.

Outputs (if model data exists):
  <cwd>/loss_plots/ALL_LR_LOSS.csv
  <cwd>/loss_plots/ALL_LR_IM_d0_LOSS.png
  <cwd>/loss_plots/ALL_LR_IM_d1_LOSS.png
  <cwd>/loss_plots/ALL_LR_IM_d2_LOSS.png
  <cwd>/loss_plots/ALL_LR_RT_d0_LOSS.png
  <cwd>/loss_plots/ALL_LR_RT_d1_LOSS.png
  <cwd>/loss_plots/ALL_LR_RT_d2_LOSS.png
  <cwd>/loss_plots/ALL_LR_FR_d0_LOSS.png
  <cwd>/loss_plots/ALL_LR_FR_d1_LOSS.png
  <cwd>/loss_plots/ALL_LR_FR_d2_LOSS.png
EOF
}

S3DIR=""
DIANN_VERSION=""
PLOT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --s3dir)
      shift
      S3DIR="${1:-}"
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

[[ -n "$S3DIR" ]] || { echo "ERROR: --s3dir is required" >&2; exit 1; }
[[ -d "$S3DIR" ]] || { echo "ERROR: Missing directory: $S3DIR" >&2; exit 1; }
PLOT_DIR="${LOSS_PLOTS_DIR:-${S3DIR}/loss_plots}"
mkdir -p "$PLOT_DIR"

if [[ -n "$DIANN_VERSION" ]]; then
  DIANN_VERSION="${DIANN_VERSION#diann-}"
fi

python3 - "$S3DIR" "$DIANN_VERSION" "$PLOT_DIR" <<'PY'
import csv
import pathlib
import re
import sys

s3dir = pathlib.Path(sys.argv[1])
diann_version = sys.argv[2] if len(sys.argv) > 2 else ""

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit(f"Matplotlib is required for plotting ({exc})")

pattern = re.compile(r"^LR_(?P<lr>.+)_(?P<model>IM|RT|FR)_LOSS\.csv$")
data = {"IM": {}, "RT": {}, "FR": {}}
rows_all = []

for csv_path in sorted(s3dir.glob("lr_*/LR_*_LOSS.csv")):
    m = pattern.match(csv_path.name)
    if not m:
        continue
    lr_tag = m.group("lr")
    model = m.group("model")
    curves = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        has_distill = "distill" in (reader.fieldnames or [])
        for row in reader:
            distill = int(row.get("distill", "0")) if has_distill else 0
            percent = float(row["percent"])
            loss = float(row["loss"])
            if distill not in curves:
                curves[distill] = ([], [])
            x, y = curves[distill]
            x.append(percent)
            y.append(loss)
            rows_all.append((lr_tag, model, distill, percent, loss))
    for distill, curve in curves.items():
        data[model][(lr_tag, distill)] = curve

if not rows_all:
    raise SystemExit(f"No LR loss CSV files found under {s3dir}/lr_*")

rows_all.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
combined_csv = pathlib.Path(sys.argv[3]) / "ALL_LR_LOSS.csv"
with combined_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["lr_tag", "model", "distill", "percent", "loss"])
    for row in rows_all:
        w.writerow(row)

created = 0
for model in ("IM", "RT", "FR"):
    curves = data[model]
    if not curves:
        continue

    distill_levels = sorted({distill for (_lr_tag, distill) in curves})
    for distill_level in distill_levels:
        distill_curves = {
            lr_tag: curve
            for (lr_tag, distill), curve in curves.items()
            if distill == distill_level
        }
        if not distill_curves:
            continue

        plt.figure(figsize=(7.5, 4.5))
        lr_tags = sorted(distill_curves)
        color_map = plt.get_cmap("tab10", max(len(lr_tags), 1))
        lr_colors = {lr_tag: color_map(idx) for idx, lr_tag in enumerate(lr_tags)}

        for lr_tag in lr_tags:
            x, y = distill_curves[lr_tag]
            plt.plot(x, y, linewidth=1.3, color=lr_colors[lr_tag], label=f"LR {lr_tag.replace('p', '.')}" )

        plt.xlabel("Training progress [%]")
        plt.ylabel("Loss")
        title = f"All LR runs - {model} loss - d{distill_level}"
        if diann_version:
            title += f" (DIA-NN {diann_version})"
        plt.title(title)
        plt.grid(True, alpha=0.35)
        plt.legend(fontsize=8)
        plt.tight_layout()
        out_png = pathlib.Path(sys.argv[3]) / f"ALL_LR_{model}_d{distill_level}_LOSS.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        created += 1

if created == 0:
    raise SystemExit(f"No model-specific curves found under {s3dir}/lr_*")

print(f"Created {created} combined LR plot(s) and {combined_csv.name} in {s3dir}")
PY
