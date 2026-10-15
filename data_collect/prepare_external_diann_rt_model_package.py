#!/usr/bin/env python3
"""Archive one DIA-NN RT sweep checkpoint as an immutable external-model package.

The package is the hand-off contract between GPU host GPU fine-tuning and a
analysis host/cluster Step-4 + Step-5 search.  It deliberately contains substantially
more provenance than the search itself needs, so an external search remains
auditable after source paths or code have changed.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP = Path(
    "./shared_data/diann_fintune_out/"
    "diann_trainfit_broad_screen_v2"
)
DEFAULT_PACKAGE_ROOT = Path(
    "./shared_data/external_models/diann"
)
DEFAULT_MODEL_DIR = Path(os.environ.get(
    "DIANN_MODEL_DIR", ROOT.parent / "diann_sc/DIA-NN-2.5.0/diann-2.5.0/models"
))
DEFAULT_DIANN_ROOT = Path(os.environ.get("DIANN_FINETUNE_ROOT", ROOT.parent / "diann-fintune"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP,
                        help="Sweep root containing sweep_manifest.json.")
    parser.add_argument(
        "--run-dir", type=Path,
        help="Export this exact completed DIA-NN run. This bypasses manifest selection."
    )
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--target", help="Match metadata.target (or legacy metadata.mode).")
    parser.add_argument("--arm", help="Match metadata.arm, e.g. weighting or oversampling.")
    parser.add_argument("--profile", help="Match metadata.profile, e.g. W5 or O9.")
    parser.add_argument("--loss", help="Match metadata.loss.")
    parser.add_argument("--lr", type=float, help="Match metadata.lr numerically.")
    parser.add_argument(
        "--metadata", action="append", default=[], metavar="KEY=VALUE",
        help="Additional exact metadata selector; repeatable."
    )
    # Keep the old command-line contract usable without making its directory
    # layout part of the exporter contract.
    parser.add_argument("--mode", help="Deprecated alias for --target.")
    parser.add_argument("--policy", help="Match legacy/current metadata.policy.")
    parser.add_argument("--lr-level", help="Match legacy metadata.lr_level.")
    parser.add_argument("--seed", type=int, help="Use this seed instead of choosing the validation medoid.")
    parser.add_argument(
        "--checkpoint-selection",
        help="Export this named checkpoint-grid selection, e.g. 0_100. Defaults to the sweep primary selection when present.",
    )
    parser.add_argument("--package-name", help="Stable package name; default is derived from the selected run.")
    parser.add_argument("--diann-model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--diann-finetune-root", type=Path, default=DEFAULT_DIANN_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Resolve and validate the selection without writing a package.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(*args: str, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def copy(source: Path, destination: Path, copied: list[dict]) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    package_root = next(
        ancestor.parent for ancestor in destination.parents
        if ancestor.name in {"model", "provenance"}
    )
    copied.append({
        "package_path": str(destination.relative_to(package_root)),
        "source_path": str(source.resolve()),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    })


def export_cpu_torchscript(source: Path, destination: Path, copied: list[dict]) -> None:
    """Copy a checkpoint with CPU tensor storage for CPU-only DIA-NN.

    ``map_location='cpu'`` is not available to the DIA-NN command-line model
    loader.  Therefore the package must contain weights serialized on CPU, in
    addition to DIA-NN's scripted ``model.device`` attribute being ``'cpu'``.
    """
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = torch.jit.load(str(source), map_location="cpu").cpu()
    if hasattr(model, "device"):
        model.device = "cpu"
    torch.jit.save(model, str(destination))

    # Intentionally load without map_location: this mirrors the CPU-only
    # DIA-NN loader and catches accidental CUDA tensor storage immediately.
    verified = torch.jit.load(str(destination))
    devices = {parameter.device.type for parameter in verified.parameters()}
    if devices != {"cpu"} or getattr(verified, "device", "cpu") != "cpu":
        raise RuntimeError(
            f"CPU-only TorchScript validation failed for {destination}: "
            f"parameters={sorted(devices)}, "
            f"scripted_device={getattr(verified, 'device', '<absent>')!r}"
        )
    package_root = next(
        ancestor.parent for ancestor in destination.parents
        if ancestor.name in {"model", "provenance"}
    )
    copied.append({
        "package_path": str(destination.relative_to(package_root)),
        "source_path": str(source.resolve()),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "export_transform": "TorchScript tensors remapped to CPU; scripted device set to cpu",
    })


def validate_packaged_dictionary(source: Path, packaged: Path, model: Path) -> dict[str, int | str]:
    """Check that the exact run dictionary accompanies the packaged model."""
    if sha256(source) != sha256(packaged):
        raise RuntimeError(f"Packaged dictionary differs from selected run dictionary: {packaged}")
    token_ids: list[int] = []
    for line_number, line in enumerate(packaged.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            _, raw_id = line.rsplit("\t", 1)
            token_ids.append(int(raw_id))
        except ValueError as exc:
            raise RuntimeError(f"Invalid token dictionary row {line_number} in {packaged}") from exc
    if not token_ids:
        raise RuntimeError(f"Token dictionary is empty: {packaged}")
    loaded_model = torch.jit.load(str(model))
    embedding_rows = int(loaded_model.embedding.weight.shape[0])
    if min(token_ids) < 0 or max(token_ids) >= embedding_rows:
        raise RuntimeError(
            f"Dictionary token IDs [{min(token_ids)}, {max(token_ids)}] do not fit "
            f"the model embedding with {embedding_rows} rows"
        )
    return {
        "source_sha256": sha256(source),
        "packaged_sha256": sha256(packaged),
        "tokens": len(token_ids),
        "max_token_id": max(token_ids),
        "embedding_rows": embedding_rows,
    }


def parse_metadata_selectors(values: list[str]) -> dict[str, str]:
    selectors: dict[str, str] = {}
    for value in values:
        key, separator, expected = value.partition("=")
        if not separator or not key or not expected:
            raise ValueError(f"--metadata must use KEY=VALUE, got {value!r}")
        if key in selectors and selectors[key] != expected:
            raise ValueError(f"Conflicting --metadata selector for {key!r}")
        selectors[key] = expected
    return selectors


def metadata_value(metadata: dict, key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def selector_map(args: argparse.Namespace) -> dict[str, str]:
    if args.target and args.mode and args.target != args.mode:
        raise ValueError("--target and deprecated --mode disagree")
    selectors = parse_metadata_selectors(args.metadata)
    built_in = {
        "target": args.target or args.mode,
        "arm": args.arm,
        "profile": args.profile,
        "loss": args.loss,
        "policy": args.policy,
        "lr_level": args.lr_level,
    }
    for key, value in built_in.items():
        if value is None:
            continue
        if key in selectors and selectors[key] != str(value):
            raise ValueError(f"Conflicting selector for {key!r}")
        selectors[key] = str(value)
    return selectors


def job_run_dir(job: dict) -> Path | None:
    for item in job.get("completion_files", []):
        path = Path(item)
        if path.parent.name == "validation_metrics" and path.name.startswith("checkpoint_"):
            return path.parent.parent
        if path.name.endswith("metrics.json"):
            return path.parent
    completion = job.get("completion_files", [])
    return Path(completion[0]).parent if completion else None


def validation_metric(metrics_path: Path) -> tuple[str, float]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if isinstance(metrics.get("source_balanced_raw_mae"), (int, float)):
        return "source_balanced_raw_mae", float(metrics["source_balanced_raw_mae"])
    result = metrics.get("result", {})
    finetuned = result.get("finetuned_metrics", {}) if isinstance(result, dict) else {}
    if isinstance(finetuned.get("mae"), (int, float)):
        return "finetuned_metrics.mae", float(finetuned["mae"])
    raise ValueError(
        f"Cannot select a validation medoid from {metrics_path}; expected "
        "source_balanced_raw_mae or result.finetuned_metrics.mae"
    )


def command_option(command: list[str], option: str) -> str | None:
    values = [command[index + 1] for index, item in enumerate(command[:-1]) if item == option]
    if len(values) > 1:
        raise ValueError(f"Manifest command contains {option} more than once")
    return values[0] if values else None


def infer_manifest_metadata(job: dict) -> dict:
    """Normalize new metadata-rich and older path-only manifest jobs."""
    metadata = dict(job.get("metadata", {}))
    commands = job.get("commands", [])
    training = commands[0] if commands and isinstance(commands[0], list) else []
    for option, key in (("--seed", "seed"), ("--loss", "loss"), ("--trainable", "policy")):
        value = command_option(training, option)
        if value is not None and key not in metadata:
            metadata[key] = value
    learning_rate = command_option(training, "--learning-rate")
    if learning_rate is not None and "lr" not in metadata:
        try:
            metadata["lr"] = float(learning_rate)
        except ValueError:
            metadata["lr"] = learning_rate
    train_split = command_option(training, "--train-split")
    if train_split and "target" not in metadata and "mode" not in metadata:
        parts = Path(train_split).parts
        if "train" in parts:
            index = parts.index("train")
            if index > 0:
                metadata["target"] = parts[index - 1]

    # Older sweeps have no metadata but use target/policy/lr-level/seed IDs.
    job_parts = str(job.get("id", "")).split("/")
    if len(job_parts) >= 4 and job_parts[-1].startswith("seed_"):
        metadata.setdefault("target", job_parts[0])
        metadata.setdefault("policy", job_parts[1])
        metadata.setdefault("lr_level", job_parts[2])
        metadata.setdefault("seed", job_parts[-1].removeprefix("seed_"))
    return metadata


def selected_artifacts(run_dir: Path, checkpoint_selection: str | None) -> tuple[Path, Path]:
    """Resolve legacy root artifacts or a named checkpoint-grid selection."""
    if checkpoint_selection is not None:
        metrics = run_dir / "validation_metrics" / f"checkpoint_{checkpoint_selection}.json"
        if not metrics.is_file():
            raise FileNotFoundError(f"Checkpoint metrics are missing: {metrics}")
        record = json.loads(metrics.read_text(encoding="utf-8"))
        checkpoint_value = record.get("checkpoint_path")
        checkpoint = Path(checkpoint_value) if isinstance(checkpoint_value, str) else Path()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Selected checkpoint is missing: {checkpoint}")
        return checkpoint, metrics
    metrics = run_dir / "validation_metrics.json"
    checkpoint = run_dir / "rt.d0.finetuned.pt"
    if not metrics.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(
            f"Run has no legacy validation_metrics.json/rt.d0.finetuned.pt pair: {run_dir}; "
            "pass --checkpoint-selection for a checkpoint-grid sweep"
        )
    return checkpoint, metrics


def candidate_from_job(job: dict, checkpoint_selection: str | None) -> dict | None:
    run_dir = job_run_dir(job)
    if run_dir is None:
        return None
    try:
        checkpoint, metrics_path = selected_artifacts(run_dir, checkpoint_selection)
    except FileNotFoundError:
        return None
    metric_name, metric_value = validation_metric(metrics_path)
    metadata = infer_manifest_metadata(job)
    seed = metadata.get("seed")
    if seed is None and run_dir.name.startswith("seed_"):
        seed = run_dir.name.removeprefix("seed_")
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        return None
    return {
        "job": job, "metadata": metadata, "run_dir": run_dir,
        "checkpoint": checkpoint, "metrics_path": metrics_path, "seed": seed,
        "checkpoint_selection": checkpoint_selection,
        "selection_metric": metric_name, "selection_value": metric_value,
    }


def candidate_matches(candidate: dict, selectors: dict[str, str], lr: float | None) -> bool:
    metadata = candidate["metadata"]
    for key, expected in selectors.items():
        actual = metadata_value(metadata, key)
        # Older manifests called the target coordinate "mode".
        if key == "target" and actual is None:
            actual = metadata_value(metadata, "mode")
        if actual != expected:
            return False
    if lr is not None:
        try:
            if abs(float(metadata["lr"]) - lr) > max(1e-12, abs(lr) * 1e-9):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def configuration_key(candidate: dict) -> str:
    metadata = {key: value for key, value in candidate["metadata"].items() if key != "seed"}
    return json.dumps(metadata, sort_keys=True, default=str)


def resolve_manifest_candidate(sweep: Path, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    manifest_path = sweep / "sweep_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Sweep manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selectors = selector_map(args)
    checkpoint_selection = args.checkpoint_selection or manifest.get("primary_selection")
    candidates = [
        candidate for job in manifest.get("jobs", [])
        if (candidate := candidate_from_job(job, checkpoint_selection)) is not None
        and candidate_matches(candidate, selectors, args.lr)
    ]
    if not candidates:
        detail = ", ".join(f"{key}={value}" for key, value in selectors.items()) or "no selectors"
        raise FileNotFoundError(f"No completed DIA-NN run matches {detail}")
    configurations = {configuration_key(candidate) for candidate in candidates}
    if len(configurations) != 1:
        examples = []
        for candidate in candidates:
            metadata = candidate["metadata"]
            example = ", ".join(
                f"{key}={metadata[key]}" for key in ("arm", "target", "mode", "policy", "profile", "loss", "lr")
                if key in metadata
            )
            if example and example not in examples:
                examples.append(example)
        raise ValueError(
            "Selection matches multiple configurations. Add --arm/--target/--profile/--loss/--lr "
            f"or --metadata KEY=VALUE. Matches include: {'; '.join(examples[:8])}"
        )
    metric_names = {candidate["selection_metric"] for candidate in candidates}
    if len(metric_names) != 1:
        raise ValueError(f"Selected seed runs use incompatible medoid metrics: {sorted(metric_names)}")
    candidates.sort(key=lambda candidate: candidate["seed"])
    if args.seed is not None:
        selected = next((candidate for candidate in candidates if candidate["seed"] == args.seed), None)
        if selected is None:
            raise ValueError(f"Seed {args.seed} is not available; found {[candidate['seed'] for candidate in candidates]}")
        return selected, candidates
    mean = sum(candidate["selection_value"] for candidate in candidates) / len(candidates)
    selected = min(
        candidates,
        key=lambda candidate: (abs(candidate["selection_value"] - mean), candidate["seed"]),
    )
    return selected, candidates


def resolve_direct_candidate(run_dir: Path, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    run_dir = run_dir.resolve()
    checkpoint, metrics_path = selected_artifacts(run_dir, args.checkpoint_selection)
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    job = None
    # A direct run path must still be useful for a manifest-based sweep.  When
    # possible, recover its compact configuration metadata without requiring
    # callers to repeat it on the command line.
    manifest_path = args.sweep_dir.resolve() / "sweep_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for possible_job in manifest.get("jobs", []):
            possible_run_dir = job_run_dir(possible_job)
            if possible_run_dir is not None and possible_run_dir.resolve() == run_dir:
                job = possible_job
                metadata = infer_manifest_metadata(possible_job)
                break
    seed = metadata.get("seed", run_dir.name.removeprefix("seed_"))
    metric_name, metric_value = validation_metric(metrics_path)
    candidate = {
        "job": job, "metadata": metadata, "run_dir": run_dir,
        "checkpoint": checkpoint, "metrics_path": metrics_path, "seed": int(seed),
        "checkpoint_selection": args.checkpoint_selection,
        "selection_metric": metric_name, "selection_value": metric_value,
    }
    return candidate, [candidate]


def register_model(package_root: Path, row: dict[str, str]) -> None:
    """Append one package to the family registry, safely across simultaneous exports."""
    path = package_root / "models_list.csv"
    fields = list(row)
    package_root.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        existing = list(csv.DictReader(handle))
        if any(item.get("model_id") == row["model_id"] for item in existing):
            raise FileExistsError(f"Model ID is already registered: {row['model_id']}")
        handle.seek(0, os.SEEK_END)
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not existing:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)


def copy_if_file(source: Path, destination: Path, copied: list[dict]) -> bool:
    if not source.is_file():
        return False
    copy(source, destination, copied)
    return True


def safe_label(value: object) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in str(value))


def dataset_provenance(run_dir: Path, metadata: dict) -> tuple[Path | None, Path | None, Path | None]:
    """Recover common train/validation/alignment artifacts without assuming a run layout."""
    metadata_path = run_dir / "run_metadata.json"
    run_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    raw_input = run_metadata.get("input")
    if not isinstance(raw_input, str):
        return None, None, None
    training_input = Path(raw_input)
    target = metadata.get("target", metadata.get("mode"))
    if not isinstance(target, str):
        return training_input if training_input.is_file() else None, None, None
    for ancestor in training_input.parents:
        alignment = ancestor / target / "rt_alignment.json"
        validation = ancestor / target / "validation" / "diann_rt_input" / "rt_validation_data.tsv"
        if alignment.is_file() or validation.is_file():
            return (
                training_input if training_input.is_file() else None,
                validation if validation.is_file() else None,
                alignment if alignment.is_file() else None,
            )
    return training_input if training_input.is_file() else None, None, None


def main() -> None:
    args = parse_args()
    if args.run_dir is not None and (args.seed is not None or args.target or args.mode or args.arm or args.profile or args.loss or args.lr is not None or args.metadata):
        raise ValueError("--run-dir selects one exact run and cannot be combined with selectors or --seed")
    sweep = args.sweep_dir.resolve()
    manifest_path = sweep / "sweep_manifest.json"
    if args.run_dir is not None:
        selected, seed_candidates = resolve_direct_candidate(args.run_dir, args)
    else:
        selected, seed_candidates = resolve_manifest_candidate(sweep, args)
    run_dir = selected["run_dir"]
    seed = selected["seed"]
    metadata = selected["metadata"]
    checkpoint = selected["checkpoint"]
    dictionary = run_dir / "dict.txt"
    if not dictionary.is_file():
        raise FileNotFoundError(f"Selected run has no token dictionary: {dictionary}")
    checkpoint_sha256 = sha256(checkpoint)
    model_id = f"diannrt_{checkpoint_sha256[:12]}"
    target_label = metadata.get("target", metadata.get("mode", "target"))
    profile_label = metadata.get("profile", metadata.get("policy", "profile"))
    lr_label = metadata.get("lr", metadata.get("lr_level", "lr"))
    package_name = args.package_name or (
        f"{datetime.now().date().isoformat()}_{model_id}_{safe_label(target_label)}_"
        f"{safe_label(profile_label)}_{safe_label(lr_label)}_medoid_seed{seed}_v1"
    )
    if not all(character.isalnum() or character in "._-" for character in package_name):
        raise ValueError("Package name may contain only letters, digits, '.', '_' and '-'")
    package_root = args.package_root.resolve()
    package = package_root / package_name
    if package.exists():
        raise FileExistsError(f"Refusing to overwrite immutable package: {package}")
    staging = package_root / f".{package_name}.staging.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")

    selection_metric = selected["selection_metric"]
    seed_mean = sum(item["selection_value"] for item in seed_candidates) / len(seed_candidates)
    seed_selection_method = (
        f"explicit validation-selected seed {seed}"
        if args.seed is not None
        else f"validation medoid: {selection_metric} nearest the selected seeds' mean"
    )
    if args.dry_run:
        print(json.dumps({
            "run_dir": str(run_dir), "seed": seed, "selection_metric": selection_metric,
            "checkpoint": str(checkpoint), "checkpoint_selection": selected["checkpoint_selection"],
            "selected_value": selected["selection_value"], "seed_mean": seed_mean,
            "available_seeds": [item["seed"] for item in seed_candidates],
            "configuration": metadata, "package": str(package),
        }, indent=2, default=str))
        return

    training_input, validation_input, alignment = dataset_provenance(run_dir, metadata)
    job = selected["job"]
    copied: list[dict] = []
    payload = staging / "model_package"
    try:
        # Artifacts required by the future Step-4 call.
        export_cpu_torchscript(checkpoint, payload / "model/rt.d0.finetuned.pt", copied)
        copy(dictionary, payload / "model/dict.txt", copied)
        # Preserve the last epoch too: it makes checkpoint-selection auditable.
        export_cpu_torchscript(run_dir / "rt.d0.last.pt", payload / "model/rt.d0.last.pt", copied)
        dictionary_validation = validate_packaged_dictionary(
            dictionary,
            payload / "model/dict.txt",
            payload / "model/rt.d0.finetuned.pt",
        )

        # Per-run provenance and any recoverable labelled inputs.  Individual
        # sweep wrappers differ, so only the checkpoint/dictionary above are
        # mandatory export inputs.
        for name in ("run_metadata.json", "validation_metrics.json", "training_history.csv", "train_split.tsv", "validation_split.tsv", "checkpoint_selection.json"):
            copy_if_file(run_dir / name, payload / f"provenance/run/{name}", copied)
        copy(selected["metrics_path"], payload / "provenance/run/selected_validation_metrics.json", copied)
        if training_input is not None:
            copy(training_input, payload / "provenance/dataset/rt_train_data.tsv", copied)
        if validation_input is not None:
            copy(validation_input, payload / "provenance/dataset/rt_validation_data.tsv", copied)
        if alignment is not None:
            copy(alignment, payload / "provenance/dataset/rt_alignment.json", copied)

        # Whole-study records are optional: every analysis writer has its own
        # filenames, but manifests and common ranking/summary outputs remain
        # useful provenance when present.
        copy_if_file(manifest_path, payload / "provenance/sweep/sweep_manifest.json", copied)
        for name in (
            "validation_summary.csv", "configuration_mean_summary.csv", "heavy_macro_ranking.csv",
            "guardrail_pass_candidates.csv", "matched_control_deltas.csv", "README.md",
        ):
            source = sweep / (name if name == "README.md" else f"analysis/{name}")
            copy_if_file(source, payload / f"provenance/sweep/{name}", copied)

        # Record the base model and trainer used to make the checkpoint.
        copy(args.diann_model_dir.resolve() / "rt.d0.pt", payload / "provenance/base_model/rt.d0.pt", copied)
        copy(args.diann_model_dir.resolve() / "dict.txt", payload / "provenance/base_model/dict.txt", copied)
        trainer = args.diann_finetune_root.resolve() / "train_carafe_rt.py"
        copy_if_file(trainer, payload / "provenance/software/train_carafe_rt.py", copied)
        if job is not None:
            commands_dir = payload / "provenance/commands"
            commands_dir.mkdir(parents=True)
            for index, command in enumerate(job.get("commands", []), start=1):
                (commands_dir / f"command_{index:02d}_argv.json").write_text(
                    json.dumps(command, indent=2) + "\n", encoding="utf-8"
                )

        environment = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "prepared_on_host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "diann_finetune_git_commit": run_command("git", "rev-parse", "HEAD", cwd=args.diann_finetune_root),
            "diann_finetune_git_status": run_command("git", "status", "--short", cwd=args.diann_finetune_root),
            "gpu_snapshot": run_command("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
        }
        (payload / "provenance/environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
        model_manifest = {
            "package_schema": "external_diann_rt_model_package/v1",
            "package_name": package_name,
            "model_id": model_id,
            "purpose": "Provisional external RT model for Step-4 + Step-5 integration smoke tests; not a final model-selection result.",
            "model_family": "DIA-NN",
            "diann_version_contract": "2.5.0",
            "source_sweep": str(sweep),
            "source_run": str(run_dir),
            "configuration": {**metadata, "seed": seed, "checkpoint_selection": selected["checkpoint_selection"]},
            "seed_selection": {
                "method": seed_selection_method,
                "metric": selection_metric,
                "mean": seed_mean,
                "selected_value": selected["selection_value"],
                "candidates": [{"seed": item["seed"], "value": item["selection_value"]} for item in seed_candidates],
            },
            "step4_contract": {
                "rt_model": "model_package/model/rt.d0.finetuned.pt",
                "tokens": "model_package/model/dict.txt",
                "fragment_model": "Must be supplied by the existing experiment; this package intentionally changes RT only.",
            },
            "cluster_compatibility": {
                "checkpoint_tensor_storage": "cpu",
                "scripted_model_device": "cpu",
                "dictionary_validation": dictionary_validation,
            },
            "files": copied,
        }
        (payload / "model_manifest.json").write_text(json.dumps(model_manifest, indent=2) + "\n", encoding="utf-8")
        readme = f"""# External DIA-NN RT model: `{package_name}`

This model directory is the complete hand-off contract between GPU host GPU fine-tuning and analysis host/cluster **Step 4 + Step 5** searches. `model_package/` is immutable after creation. `searches/` is intentionally empty now and receives one subdirectory per dataset/search only when DIA-NN runs.

## Selected run

- Sweep: `{sweep}`
- Configuration: `{json.dumps(metadata, sort_keys=True, default=str)}` / seed `{seed}`
- Stable model ID: `{model_id}` (first 12 hexadecimal characters of the fine-tuned RT-checkpoint SHA-256).
- Seed choice: `{seed_selection_method}` by `{selection_metric}`; selected value `{selected['selection_value']:.6f}`, mean `{seed_mean:.6f}`.
- DIA-NN contract: 2.5.0.

## Canonical locations

The shared storage is canonical. This one model directory contains both its immutable package and all later searches:

```text
<SHARED_DATA_ROOT>/
  external_models/diann/{package_name}/
    model_package/                               # checkpoint, dictionary, provenance, checksums
    searches/                                    # initially empty
      <dataset_or_experiment_id>_<search_id>/
        04_refined_lib/  05_quant/  logs/  comparison/
```

One model may be searched against one or up to four datasets. The search launcher creates a direct child in `searches/` only for an actually started dataset-specific run.

## Use on analysis host

analysis host should read this model directory directly from its shared storage mount. The future external-model launcher copies only `model_package/` to job-local scratch, runs Step 4 and Step 5 there, then writes results back into this model's `searches/` folder. Do **not** copy results into the original pipeline's `04_refined_lib` or `05_quant` directories.

Before launching, verify the package from its root:

```bash
cd model_package && sha256sum -c checksums.sha256
```

The launcher must receive `model_package/model/rt.d0.finetuned.pt` and `model_package/model/dict.txt`. It derives FASTA, modification settings, raw files and the unchanged fragment model from the target experiment. This package changes RT only.

## Use on a cluster without the share mounted

First make a local, read-only job copy. Set the two root variables to the machine-specific shared mount and local scratch path:

```bash
MODEL_ID={package_name}
MODEL_STORE=<SHARED_DATA_ROOT>/external_models
LOCAL_ROOT=<CLUSTER_SCRATCH>/external_diann_rt

rsync -a --checksum --info=progress2 \\
  "$MODEL_STORE/$MODEL_ID/model_package/" "$LOCAL_ROOT/models/$MODEL_ID/model_package/"
(cd "$LOCAL_ROOT/models/$MODEL_ID/model_package" && sha256sum -c checksums.sha256)
```

Run Step 4 and Step 5 with the local copy, writing **all** outputs below one local search directory. After a successful or failed run, copy the complete directory back without deleting prior shared results:

```bash
SEARCH_ROOT="$MODEL_STORE/$MODEL_ID/searches"
EXPERIMENT_ID=<existing_experiment_or_dataset_id>
SEARCH_ID=<unique_search_id>

rsync -a --checksum --info=progress2 \\
  "$LOCAL_ROOT/searches/$EXPERIMENT_ID/$SEARCH_ID/" \\
  "$SEARCH_ROOT/${{EXPERIMENT_ID}}_${{SEARCH_ID}}/"
```

Each parallel search needs its own `SEARCH_ID` and local scratch directory. Never use `--delete` for either transfer.

## Scientific status

This package is for an integration smoke test and must not be used to select hyperparameters while the LR-refinement sweep is unfinished.
"""
        (staging / "README.md").write_text(readme, encoding="utf-8")
        # Write this last: it covers every immutable payload file, but not itself.
        checksums = []
        for path in sorted(item for item in payload.rglob("*") if item.is_file()):
            checksums.append(f"{sha256(path)}  {path.relative_to(payload)}")
        (payload / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
        (staging / "searches").mkdir()
        package_root.mkdir(parents=True, exist_ok=True)
        staging.rename(package)
        register_model(package_root, {
            "model_id": model_id,
            "created_date": datetime.now().date().isoformat(),
            "package_name": package_name,
            "relative_path": package_name,
            "model_family": "DIA-NN",
            "target_space": str(target_label),
            "trainable_policy": str(metadata.get("policy", metadata.get("arm", "unknown"))),
            "lr_level": str(lr_label),
            "selected_seed": str(seed),
            "source_balanced_raw_mae": f"{selected['selection_value']:.9f}",
            "checkpoint_sha256": checkpoint_sha256,
            "source_sweep": str(sweep),
            "source_run": str(run_dir),
            "status": "provisional_integration_smoketest",
        })
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"package": str(package), "selected_seed": seed, "files": len(copied)}, indent=2))


if __name__ == "__main__":
    main()
