#!/usr/bin/env python3
"""Freeze validation-selected Carafe2 Heavy-Dimethyl finalists for final testing.

The script reads only the sweep manifest and its validation analysis.  It
never reads the test partition.  It writes a frozen candidate-selection
manifest plus a resumable evaluation manifest; the latter is the sole place
that references the final test partition.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET_DEFAULT = ROOT / "trainingdata/finalrt_4source_trainfit15_validation10_v1"
CARAFE_PYTHON_DEFAULT = Path("python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        help="Defaults to <sweep-dir>/test and must not already exist.",
    )
    parser.add_argument(
        "--top-per-target", type=int, default=1,
        help="Validation finalists per target space (default: 1; keep this small).",
    )
    parser.add_argument(
        "--selection-metric", default="heavy_macro_source_balanced_mae_mean",
        help="Validation metric to minimize when selecting finalists.",
    )
    parser.add_argument(
        "--carafe-python",
        default=os.environ.get(
            "CARAFE2_PYTHON",
            str(CARAFE_PYTHON_DEFAULT if CARAFE_PYTHON_DEFAULT.is_file() else Path(sys.executable)),
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def control_profile(arm: str) -> str:
    if arm == "weighting":
        return "W0"
    if arm == "oversampling":
        return "O0"
    raise ValueError(f"Unsupported Heavy-Dimethyl arm: {arm!r}")


def select_finalists(ranking: pd.DataFrame, *, metric: str, top_per_target: int) -> pd.DataFrame:
    if top_per_target < 1:
        raise ValueError("--top-per-target must be at least one")
    required = {"target", "arm", "profile", "loss", "lr", "guardrail_status", metric}
    missing = required - set(ranking.columns)
    if missing:
        raise ValueError(f"Validation ranking lacks required columns: {sorted(missing)}")
    candidates = ranking.loc[
        ranking["guardrail_status"].eq("PASS")
        & ~ranking["profile"].isin(("W0", "O0"))
        & ranking[metric].notna()
    ].copy()
    if candidates.empty:
        raise ValueError("No guardrail-passing validation candidates available")
    candidates = candidates.sort_values(["target", metric, "arm", "profile", "loss", "lr"])
    selected = candidates.groupby("target", sort=True, group_keys=False).head(top_per_target).copy()
    selected["validation_rank_within_target"] = selected.groupby("target").cumcount() + 1
    return selected


def source_run_by_id(sweep: dict) -> dict[str, dict]:
    result = {}
    for job in sweep.get("jobs", []):
        job_id = job.get("id")
        completion_files = job.get("completion_files", [])
        if not isinstance(job_id, str) or len(completion_files) != 1:
            raise ValueError("Sweep jobs require one id and one validation completion file")
        result[job_id] = job
    return result


def resolve_jobs(selected: pd.DataFrame, jobs_by_id: dict[str, dict]) -> list[dict]:
    selected_keys = {
        (row.arm, row.target, row.loss, float(row.lr), row.profile)
        for row in selected.itertuples(index=False)
    }
    required_keys = selected_keys | {
        (arm, target, loss, lr, control_profile(arm))
        for arm, target, loss, lr, _ in selected_keys
    }
    chosen: list[dict] = []
    for job in jobs_by_id.values():
        meta = job.get("metadata", {})
        key = (meta.get("arm"), meta.get("target"), meta.get("loss"), float(meta.get("lr")), meta.get("profile"))
        if key not in required_keys:
            continue
        role = "finalist" if key in selected_keys else "matched_control"
        copied = {**meta, "source_job_id": job["id"], "selection_role": role}
        copied["source_run_dir"] = str(Path(job["completion_files"][0]).parent)
        if not (Path(copied["source_run_dir"]) / "rt_model.pt").is_file():
            raise FileNotFoundError(
                f"Selected source run has no trained checkpoint: {copied['source_run_dir']}"
            )
        chosen.append(copied)
    expected = len(required_keys) * 3
    if len(chosen) != expected:
        raise ValueError(
            f"Expected {expected} finalist/control seed runs, found {len(chosen)}. "
            "The source sweep may be incomplete or use an unexpected design."
        )
    return sorted(chosen, key=lambda item: item["source_job_id"])


def main() -> None:
    args = parse_args()
    sweep_dir = args.sweep_dir.resolve()
    output_dir = (args.output_dir or (sweep_dir / "test")).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite test evaluation at {output_dir}; use its manifest to resume."
        )
    sweep = read_json(sweep_dir / "sweep_manifest.json")
    if sweep.get("validation_only") is not True:
        raise ValueError("Source manifest is not a validation-only Heavy-Dimethyl sweep")
    ranking_path = sweep_dir / "analysis" / "heavy_macro_ranking.csv"
    ranking = pd.read_csv(ranking_path)
    selected = select_finalists(
        ranking, metric=args.selection_metric, top_per_target=args.top_per_target,
    )
    jobs = resolve_jobs(selected, source_run_by_id(sweep))
    dataset_dir = Path(sweep.get("dataset", DATASET_DEFAULT)).resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Training dataset referenced by sweep is unavailable: {dataset_dir}")
    carafe_python = Path(args.carafe_python).resolve()
    if not carafe_python.is_file():
        raise FileNotFoundError(f"Carafe2 Python interpreter unavailable: {carafe_python}")

    output_dir.mkdir(parents=True)
    selection_columns = [
        "model", "arm", "target", "policy", "profile", "loss", "lr", "weights", "oversampling",
        "heavy_macro_source_balanced_mae_mean", "heavy_macro_source_balanced_mae_std",
        "delta_heavy_macro_source_balanced_mae", "delta_heavy_both_sb_mae",
        "delta_nonheavy_sb_mae", "delta_overall_mae", "guardrail_status",
        "validation_rank_within_target",
    ]
    available = [column for column in selection_columns if column in selected]
    selected.loc[:, available].to_csv(output_dir / "validation_selected_finalists.csv", index=False)
    selection_payload = {
        "source_sweep": str(sweep_dir),
        "source_analysis": str(ranking_path),
        "selection_partition": "validation",
        "selection_metric": args.selection_metric,
        "top_per_target": args.top_per_target,
        "finalists": selected.loc[:, available].to_dict(orient="records"),
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(selection_payload, indent=2) + "\n", encoding="utf-8"
    )

    evaluator = ROOT / "ma/data_collect/evaluate_carafe2_trainfit.py"
    manifest_jobs = []
    for meta in jobs:
        test_dir = output_dir / "runs" / Path(meta["source_job_id"])
        metrics = test_dir / "final_test_metrics.json"
        command = [
            str(carafe_python), str(evaluator), "--run-dir", meta["source_run_dir"],
            "--partition-dir", str(dataset_dir / meta["target"] / "test"),
            "--alignment", str(dataset_dir / meta["target"] / "rt_alignment.json"),
            "--output", str(metrics), "--prediction-dir", str(test_dir / "predictions"),
            "--device", "gpu", "--python", str(carafe_python),
        ]
        manifest_jobs.append({
            "id": meta["source_job_id"], "workdir": str(ROOT / "carafe2-semi"),
            "commands": [command], "completion_files": [str(metrics)], "metadata": meta,
        })
    manifest = {
        "study": f"{sweep.get('study', 'carafe2_heavy_dimethyl')}_final_test",
        "source_sweep": str(sweep_dir), "dataset": str(dataset_dir),
        "selection_manifest": str(output_dir / "selection_manifest.json"),
        "test_partition": "final", "jobs": manifest_jobs,
    }
    (output_dir / "test_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    readme = f"""# Heavy-Dimethyl final-test evaluation

`selection_manifest.json` freezes the small finalist set selected from the
validation-only ranking before any test metric is created.  `test_manifest.json`
contains the only commands that reference `<dataset>/<target>/test`.
Every test job performs prediction and metric calculation only; it never calls
the Carafe2 trainer and never changes the source sweep's model checkpoint.

## Layout

```text
test/
├── validation_selected_finalists.csv  # validation evidence used for selection
├── selection_manifest.json            # immutable selection record
├── test_manifest.json                 # resumable final-test jobs
├── runs/<source-job-id>/
│   ├── final_test_metrics.json
│   └── predictions/
└── analysis/                          # created after all final-test jobs finish
```

Selected finalists: {len(selected)} configurations × 3 seeds.  Their matched
within-arm controls are evaluated alongside them.  Do not alter the selection
manifest after test results exist; create a new `test_evaluation_*` directory
for a new validation-selected confirmation set.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir), "finalists": len(selected), "test_jobs": len(manifest_jobs),
        "selection_manifest": str(output_dir / "selection_manifest.json"),
        "test_manifest": str(output_dir / "test_manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
