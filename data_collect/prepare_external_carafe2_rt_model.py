#!/usr/bin/env python3
"""Package and register one Carafe2 RT sweep checkpoint for external inference.

This is the Carafe2 counterpart to ``prepare_external_diann_rt_model_package.py``.
It selects exactly one validation-complete sweep configuration, chooses its
validation medoid seed unless ``--seed`` is explicit, copies the checkpoint and
compact provenance into an immutable registry directory, and registers it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP = Path("./shared_data/carafe2_finetune_out/carafe2_heavy_dimethyl_rt_v1")
DEFAULT_REGISTRY = Path("./shared_data/external_models/carafe2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--run-dir", type=Path, help="Export this exact completed run.")
    parser.add_argument("--registry-root", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--target", help="Exact target coordinate, e.g. anchor_irt.")
    parser.add_argument("--arm", help="Exact sweep arm, e.g. weighting.")
    parser.add_argument("--profile", help="Exact profile, e.g. W7.")
    parser.add_argument("--loss", help="Exact loss, e.g. mae.")
    parser.add_argument("--lr", type=float, help="Exact RT learning rate.")
    parser.add_argument("--metadata", action="append", default=[], metavar="KEY=VALUE",
                        help="Additional exact manifest selector; repeatable.")
    parser.add_argument("--package-metadata", action="append", default=[], metavar="KEY=VALUE",
                        help="Provenance to retain in the exported package (not used for sweep selection); repeatable.")
    parser.add_argument("--seed", type=int, help="Export this seed instead of the validation medoid.")
    parser.add_argument("--package-name", help="Immutable direct child name below models/.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and validate without writing.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_label(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def selector_map(values: list[str], args: argparse.Namespace) -> dict[str, str]:
    selectors: dict[str, str] = {}
    for value in values:
        key, sep, expected = value.partition("=")
        if not sep or not key or not expected:
            raise ValueError(f"--metadata must be KEY=VALUE, got {value!r}")
        selectors[key] = expected
    for key in ("target", "arm", "profile", "loss"):
        value = getattr(args, key)
        if value is not None:
            if key in selectors and selectors[key] != value:
                raise ValueError(f"Conflicting selector for {key}")
            selectors[key] = value
    return selectors


def metadata_map(values: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for value in values:
        key, sep, content = value.partition("=")
        if not sep or not key or not content:
            raise ValueError(f"--package-metadata must be KEY=VALUE, got {value!r}")
        if key in metadata:
            raise ValueError(f"Duplicate --package-metadata key: {key}")
        metadata[key] = content
    return metadata


def metric(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    value = data.get("source_balanced_raw_mae")
    if not isinstance(value, (int, float)):
        raise ValueError(f"No source_balanced_raw_mae in {path}")
    return float(value)


def run_dir_from_job(job: dict) -> Path | None:
    for raw in job.get("completion_files", []):
        path = Path(raw)
        if path.name == "validation_metrics.json":
            return path.parent
    return None


def infer_metadata(job: dict) -> dict:
    """Normalise current metadata-rich and older path-only sweep manifests."""
    metadata = dict(job.get("metadata", {}))
    parts = str(job.get("id", "")).split("/")
    # LR refinement used target/policy/lr-level/seed IDs before metadata was
    # added to the heavy-dimethyl manifests.
    if len(parts) >= 4 and parts[-1].startswith("seed_"):
        metadata.setdefault("target", parts[0])
        metadata.setdefault("policy", parts[1])
        metadata.setdefault("lr_level", parts[2])
        try:
            metadata.setdefault("seed", int(parts[-1].removeprefix("seed_")))
        except ValueError:
            pass
    return metadata


def matches(metadata: dict, selectors: dict[str, str], lr: float | None) -> bool:
    if any(str(metadata.get(key)) != expected for key, expected in selectors.items()):
        return False
    if lr is None:
        return True
    try:
        return abs(float(metadata["lr"]) - lr) <= max(1e-12, abs(lr) * 1e-9)
    except (KeyError, TypeError, ValueError):
        return False


def resolve_from_manifest(sweep: Path, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    manifest_path = sweep / "sweep_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selectors = selector_map(args.metadata, args)
    candidates: list[dict] = []
    for job in manifest.get("jobs", []):
        metadata = infer_metadata(job)
        run_dir = run_dir_from_job(job)
        if run_dir is None or not matches(metadata, selectors, args.lr):
            continue
        checkpoint, metrics = run_dir / "rt_model.pt", run_dir / "validation_metrics.json"
        if not checkpoint.is_file() or not metrics.is_file():
            continue
        seed = metadata.get("seed")
        if not isinstance(seed, int):
            continue
        candidates.append({"run_dir": run_dir, "checkpoint": checkpoint, "metrics": metrics,
                           "metadata": metadata, "seed": seed, "value": metric(metrics), "job": job})
    if not candidates:
        detail = ", ".join(f"{key}={value}" for key, value in selectors.items()) or "no selectors"
        raise FileNotFoundError(f"No completed Carafe2 run matches {detail}")
    configurations = {json.dumps({key: value for key, value in item["metadata"].items() if key != "seed"},
                                 sort_keys=True) for item in candidates}
    if len(configurations) != 1:
        raise ValueError("Selection matches multiple configurations; add --target/--arm/--profile/--loss/--lr.")
    candidates.sort(key=lambda item: item["seed"])
    if args.seed is not None:
        selected = next((item for item in candidates if item["seed"] == args.seed), None)
        if selected is None:
            raise ValueError(f"Seed {args.seed} is unavailable; found {[item['seed'] for item in candidates]}")
    else:
        mean = sum(item["value"] for item in candidates) / len(candidates)
        selected = min(candidates, key=lambda item: (abs(item["value"] - mean), item["seed"]))
    return selected, candidates


def resolve_direct(run_dir: Path, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    run_dir = run_dir.resolve()
    checkpoint, metrics = run_dir / "rt_model.pt", run_dir / "validation_metrics.json"
    if not checkpoint.is_file() or not metrics.is_file():
        raise FileNotFoundError(f"Expected rt_model.pt and validation_metrics.json in {run_dir}")
    try:
        seed = int(run_dir.name.removeprefix("seed_"))
    except ValueError as error:
        raise ValueError("--run-dir must be a seed_<integer> directory") from error
    target = args.target
    if not target:
        raise ValueError("--run-dir requires --target because target-space provenance cannot be inferred safely")
    item = {"run_dir": run_dir, "checkpoint": checkpoint, "metrics": metrics,
            "metadata": {"target": target}, "seed": seed, "value": metric(metrics), "job": None}
    return item, [item]


def copy(source: Path, destination: Path, copied: list[dict]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied.append({"package_path": str(destination), "source_path": str(source.resolve()),
                   "sha256": sha256(destination), "size_bytes": destination.stat().st_size})


def copy_if_exists(source: Path, destination: Path, copied: list[dict]) -> None:
    if source.is_file():
        copy(source, destination, copied)


def main() -> None:
    args = parse_args()
    if args.run_dir and (args.arm or args.profile or args.loss or args.lr is not None or args.metadata or args.seed is not None):
        raise ValueError("--run-dir cannot be combined with configuration selectors or --seed")
    sweep = args.sweep_dir.resolve()
    package_metadata = metadata_map(args.package_metadata)
    selected, seeds = resolve_direct(args.run_dir, args) if args.run_dir else resolve_from_manifest(sweep, args)
    metadata = selected["metadata"]
    target = str(metadata["target"])
    registry_root = args.registry_root.resolve()
    models_root = registry_root / "models"
    checksum = sha256(selected["checkpoint"])
    model_id = f"carafe2rt_{checksum[:12]}"
    profile = metadata.get("profile", metadata.get("policy", "model"))
    lr = metadata.get("lr", "lr")
    package_name = args.package_name or (
        f"{datetime.now().date().isoformat()}_{model_id}_{safe_label(target)}_"
        f"{safe_label(profile)}_{safe_label(lr)}_medoid_seed{selected['seed']}_v1"
    )
    if not package_name or not all(ch.isalnum() or ch in "._-" for ch in package_name):
        raise ValueError("--package-name may contain only letters, digits, '.', '_' and '-'")
    package = models_root / package_name
    seed_mean = sum(item["value"] for item in seeds) / len(seeds)
    plan = {"package": str(package), "model_id": model_id, "target_space": target,
            "source_run": str(selected["run_dir"]), "checkpoint": str(selected["checkpoint"]),
            "checkpoint_sha256": checksum, "configuration": metadata, "package_metadata": package_metadata,
            "selected_seed": selected["seed"],
            "selection_metric": "source_balanced_raw_mae", "selected_value": selected["value"],
            "seed_mean": seed_mean, "available_seeds": [{"seed": item["seed"], "value": item["value"]} for item in seeds]}
    if args.dry_run:
        print(json.dumps(plan, indent=2, default=str))
        return
    if package.exists():
        raise FileExistsError(f"Refusing to overwrite immutable package: {package}")
    staging = models_root / f".{package_name}.staging.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Staging path already exists: {staging}")
    copied: list[dict] = []
    try:
        copy(selected["checkpoint"], staging / "rt_model.pt", copied)
        for name in ("validation_metrics.json", "model_evaluation_metrics.json", "rt_history.tsv", "training_report.html"):
            copy_if_exists(selected["run_dir"] / name, staging / "provenance" / "run" / name, copied)
        copy_if_exists(selected["run_dir"] / "training_diagnostics/data/rt_metadata.json",
                       staging / "provenance" / "run" / "rt_metadata.json", copied)
        manifest_path = sweep / "sweep_manifest.json"
        copy_if_exists(manifest_path, staging / "provenance" / "sweep" / "sweep_manifest.json", copied)
        for name in ("configuration_mean_summary.csv", "heavy_macro_ranking.csv", "guardrail_pass_candidates.csv",
                     "matched_control_deltas.csv", "README.md"):
            source = sweep / (name if name == "README.md" else f"analysis/{name}")
            copy_if_exists(source, staging / "provenance" / "sweep" / name, copied)
        if selected["job"] is not None:
            (staging / "provenance" / "commands").mkdir(parents=True, exist_ok=True)
            for number, command in enumerate(selected["job"].get("commands", []), start=1):
                (staging / "provenance" / "commands" / f"command_{number:02d}_argv.json").write_text(
                    json.dumps(command, indent=2) + "\n", encoding="utf-8")
        plan.update({"package_schema": "external_carafe2_rt_model/v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                     "seed_selection": {"method": "explicit validation-selected seed" if args.seed is not None else "validation medoid: source_balanced_raw_mae nearest seed mean",
                                        "candidates": plan["available_seeds"]}, "copied_artifacts": copied})
        (staging / "package_manifest.json").write_text(json.dumps(plan, indent=2, default=str) + "\n", encoding="utf-8")
        models_root.mkdir(parents=True, exist_ok=True)
        staging.rename(package)
        register = ROOT / "ma/data_collect/register_external_carafe2_rt_model.py"
        subprocess.run([sys.executable, str(register), "--registry-root", str(registry_root), "--model-dir", str(package),
                        "--label", package_name, "--target-space", target, "--source-run", str(selected["run_dir"])], check=True)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(json.dumps({"model_id": model_id, "package": str(package), "target_space": target,
                      "selected_seed": selected["seed"]}, indent=2))


if __name__ == "__main__":
    main()
