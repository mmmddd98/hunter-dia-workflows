from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
import re
import shlex
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
import polars as pl

from .config import AppConfig

from maplot.entrapment import (
    add_entrapment_columns,
    exclude_diann_decoys,
    entrapment_qvalue_curve,
    entrapment_qvalue_distribution,
    entrapment_summary,
    entrapment_threshold_metrics,
    exclude_entrapment,
)
from maplot import DiannAdapter, TerminiAnnotator
from .set_analysis import (
    UPSET_LEVEL_TO_ID_PARTS,
    normalize_evidence_filters,
    normalize_nterm_status,
    normalize_upset_item_level,
    prepare_upset_evidence_rows,
    prepare_upset_items,
    raw_with_termini_evidence,
)


ID_COLS = {"nterm_modif_peptide", "nterm_modif", "peptide", "protein"}
PREFERRED_REPORT_FILES = ("final_report.parquet", "final_report-first-pass.parquet")
RESTRICTED_LAYERS_RUN_PREFIX = "tuned_"
LOSS_PLOT_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".webp"}
SERVER_CACHE_QVALUE_THRESHOLD = 1.0
SERVER_CACHE_QVALUE_TAG = "qval1"
UPSET_CACHE_VERSION = "v3"
TRIPLEMIX_CACHE_VERSION = "v5"
TRIPLEMIX_REPORT_FILES = ("final_report.parquet", "final_report-first-pass.parquet")
LIVE_STALE_AFTER_SECONDS = 6 * 60 * 60
DATASET_ID_SEPARATOR = "::"


class ExperimentRepository:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def list_experiments(self) -> list[str]:
        experiments = []
        multi_version = len(self.config.pipeline_versions) > 1
        for data_root in self.config.data_roots:
            for version in self.config.pipeline_versions:
                root = self.pipeline_root_for_version(version, data_root.key)
                if not root.exists():
                    continue
                for candidate in sorted(root.iterdir()):
                    if not candidate.is_dir():
                        continue
                    if self._has_search_result(candidate):
                        experiments.append(self.dataset_id(data_root.key, version, candidate.name, multi_version=multi_version))
        return experiments

    def experiment_root(self, experiment: str) -> Path:
        version, experiment_name = self.split_dataset_id(experiment)
        return self.pipeline_root_for_version(version, self.data_root_key_for_experiment(experiment)) / experiment_name

    def pipeline_root_for_version(self, version: str, root_key: str | None = None) -> Path:
        return self.data_root(root_key or self.config.data_roots[0].key).path / version

    def data_root(self, root_key: str):
        for data_root in self.config.data_roots:
            if data_root.key == root_key:
                return data_root
        raise KeyError(f"Unknown maplot data root: {root_key}")

    def dataset_id(self, root_key: str, version: str, experiment: str, *, multi_version: bool) -> str:
        version_part = f"{version}{DATASET_ID_SEPARATOR}" if multi_version else ""
        return f"{root_key}{DATASET_ID_SEPARATOR}{version_part}{experiment}"

    def split_dataset_id(self, dataset_id: str) -> tuple[str, str]:
        parts = dataset_id.split(DATASET_ID_SEPARATOR)
        if len(parts) >= 3:
            return parts[-2], parts[-1]
        if len(parts) == 2:
            if parts[0] in {root.key for root in self.config.data_roots}:
                return self.config.pipeline_version, parts[1]
            return parts[0], parts[1]
        return self.config.pipeline_version, dataset_id

    def data_root_key_for_experiment(self, dataset_id: str) -> str:
        parts = dataset_id.split(DATASET_ID_SEPARATOR)
        if len(parts) >= 3:
            return parts[0]
        return self.config.data_roots[0].key

    def cluster_for_experiment(self, dataset_id: str) -> str:
        return self.data_root(self.data_root_key_for_experiment(dataset_id)).cluster

    def pipeline_version_for_experiment(self, experiment: str) -> str:
        return self.split_dataset_id(experiment)[0]

    def experiment_name(self, experiment: str) -> str:
        return self.split_dataset_id(experiment)[1]

    def list_loss_plots(self, experiment: str) -> list[dict]:
        root = self.experiment_root(experiment)
        tune_root = root / "03_tune_models"
        if not tune_root.exists():
            return []

        plots = []
        for path in sorted(tune_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in LOSS_PLOT_IMAGE_SUFFIXES:
                continue
            if "_LOSS" not in path.name:
                continue

            rel_to_tune = path.relative_to(tune_root)
            parts = rel_to_tune.parts
            profile = parts[0] if parts else "default"
            name = path.name
            model_match = re.search(r"_(IM|RT|FR)_", name)
            distill_match = re.search(r"_d([0-9]+)_LOSS", name)
            lr_match = re.search(r"^LR_(.+?)_(IM|RT|FR)_LOSS", name)
            combined = name.startswith("ALL_LR_")
            distill_level = int(distill_match.group(1)) if distill_match else None
            kind = "single_lr"
            if combined and distill_level and distill_level > 0:
                kind = "distillate"
            elif combined:
                kind = "combined"

            plots.append(
                {
                    "path": str(path),
                    "relative_path": path.relative_to(root).as_posix(),
                    "profile": profile,
                    "model": model_match.group(1) if model_match else "other",
                    "distill": distill_level,
                    "lr_tag": lr_match.group(1) if lr_match else None,
                    "kind": kind,
                    "filename": name,
                    "title": loss_plot_title(name, profile, kind, model_match.group(1) if model_match else "other", distill_level, lr_match.group(1) if lr_match else None),
                    "modified_ts": path.stat().st_mtime,
                }
            )

        return sorted(
            plots,
            key=lambda plot: (
                plot["profile"],
                {"combined": 0, "distillate": 1, "single_lr": 2}.get(plot["kind"], 9),
                plot["model"],
                plot["distill"] if plot["distill"] is not None else -1,
                plot["lr_tag"] or "",
                plot["filename"],
            ),
        )

    def load_loss_curves(self, experiment: str) -> pd.DataFrame:
        tune_root = self.experiment_root(experiment) / "03_tune_models"
        columns = ["profile", "lr_tag", "model", "distill", "percent", "loss"]
        if not tune_root.exists():
            return pd.DataFrame(columns=columns)

        frames = []
        for profile_dir in sorted(path for path in tune_root.iterdir() if path.is_dir()):
            combined_path = profile_dir / "loss_plots" / "ALL_LR_LOSS.csv"
            if combined_path.exists():
                try:
                    frame = pd.read_csv(combined_path)
                except (OSError, pd.errors.ParserError):
                    frame = pd.DataFrame()
                if not frame.empty:
                    frame["profile"] = profile_dir.name
                    frames.append(frame)
                    continue

            for path in sorted(profile_dir.rglob("LR_*_LOSS.csv")):
                match = re.match(r"LR_(.+?)_(IM|RT|FR)_LOSS\.csv$", path.name, flags=re.IGNORECASE)
                if not match:
                    continue
                try:
                    frame = pd.read_csv(path)
                except (OSError, pd.errors.ParserError):
                    continue
                frame["profile"] = profile_dir.name
                frame["lr_tag"] = match.group(1)
                frame["model"] = match.group(2).upper()
                frames.append(frame)

        if not frames:
            return pd.DataFrame(columns=columns)
        data = pd.concat(frames, ignore_index=True)
        for column in ("distill", "percent", "loss"):
            data[column] = pd.to_numeric(data.get(column), errors="coerce")
        data["distill"] = data["distill"].fillna(0).astype(int)
        data["lr_tag"] = data["lr_tag"].astype(str)
        data["model"] = data["model"].astype(str).str.upper()
        return (
            data.dropna(subset=["percent", "loss"])
            .loc[:, columns]
            .sort_values(["profile", "model", "distill", "lr_tag", "percent"])
            .reset_index(drop=True)
        )

    def list_live_experiments(self) -> list[str]:
        experiments = []
        multi_version = len(self.config.pipeline_versions) > 1
        for data_root in self.config.data_roots:
            for version in self.config.pipeline_versions:
                root = self.pipeline_root_for_version(version, data_root.key)
                if not root.exists():
                    continue
                for candidate in sorted(root.iterdir()):
                    experiment = self.dataset_id(data_root.key, version, candidate.name, multi_version=multi_version)
                    if candidate.is_dir() and self.is_experiment_live(experiment):
                        experiments.append(experiment)
        return experiments

    def latest_live_experiment(self) -> str | None:
        experiments = self.list_live_experiments()
        if not experiments:
            return None
        return max(experiments, key=lambda experiment: self.live_summary(experiment).get("last_activity_ts", 0))

    def live_summary(self, experiment: str) -> dict:
        root = self.experiment_root(experiment)
        events = self.load_progress_events(experiment)
        logs = self.list_ram_logs(experiment)
        pid_status = self.run_pid_status(experiment)
        latest_progress = max((event["timestamp_dt"] for event in events if event["timestamp_dt"] is not None), default=None)
        latest_ram = None
        latest_ram_label = None
        for log in logs:
            ram_df = self.load_ram_log(log["path"])
            if ram_df.empty:
                continue
            current = ram_df["timestamp"].max()
            if latest_ram is None or current > latest_ram:
                latest_ram = current
                latest_ram_label = log.get("short_label") or log.get("label")
        latest_file_ts = max((path.stat().st_mtime for path in root.rglob("*") if path.is_file()), default=root.stat().st_mtime if root.exists() else 0)
        last_activity_ts = max(
            [
                latest_file_ts,
                latest_progress.timestamp() if latest_progress is not None else 0,
                latest_ram.timestamp() if latest_ram is not None else 0,
            ]
        )
        current_step = current_progress_step(events)
        stale_live = pid_status["running"] and live_activity_is_stale(last_activity_ts)
        status = "running" if pid_status["running"] and not stale_live else "complete/idle" if events else "unknown"
        if stale_live:
            status = "stopped/idle"
        if not pid_status["running"] and any(event["state"] == "FAIL" for event in events):
            status = "failed"
        return {
            "experiment": experiment,
            "root": str(root),
            "status": status,
            "pid": pid_status["pid"],
            "pid_running": pid_status["running"] and not stale_live,
            "pid_reason": pid_status["reason"],
            "is_live": pid_status["running"] and not stale_live,
            "stale_live": stale_live,
            "current_step": current_step,
            "event_count": len(events),
            "ram_log_count": len(logs),
            "started_at": events[0]["timestamp"] if events else None,
            "last_progress_at": latest_progress.isoformat(timespec="seconds") if latest_progress is not None else None,
            "last_ram_at": latest_ram.isoformat(timespec="seconds") if latest_ram is not None else None,
            "last_ram_log": latest_ram_label,
            "last_activity_ts": last_activity_ts,
            "last_activity": datetime.fromtimestamp(last_activity_ts, tz=timezone.utc).astimezone().isoformat(timespec="seconds") if last_activity_ts else None,
        }

    def quick_live_summary(self, experiment: str) -> dict:
        events = self.load_progress_events(experiment)
        pid_status = self.run_pid_status(experiment)
        latest_progress = max((event["timestamp_dt"] for event in events if event["timestamp_dt"] is not None), default=None)
        last_activity_ts = latest_progress.timestamp() if latest_progress is not None else 0
        stale_live = pid_status["running"] and live_activity_is_stale(last_activity_ts) if last_activity_ts else False
        current_step = current_progress_step(events)
        status = "running" if pid_status["running"] and not stale_live else "complete/idle" if events else "unknown"
        if stale_live:
            status = "stopped/idle"
        if not pid_status["running"] and any(event["state"] == "FAIL" for event in events):
            status = "failed"
        return {
            "experiment": experiment,
            "status": status,
            "pid": pid_status["pid"],
            "pid_running": pid_status["running"] and not stale_live,
            "pid_reason": pid_status["reason"],
            "is_live": pid_status["running"] and not stale_live,
            "stale_live": stale_live,
            "current_step": current_step,
            "event_count": len(events),
            "started_at": events[0]["timestamp"] if events else None,
            "last_progress_at": latest_progress.isoformat(timespec="seconds") if latest_progress is not None else None,
            "last_activity_ts": last_activity_ts,
            "last_activity": datetime.fromtimestamp(last_activity_ts, tz=timezone.utc).astimezone().isoformat(timespec="seconds") if last_activity_ts else None,
        }

    def is_experiment_live(self, experiment: str) -> bool:
        return bool(self.live_summary(experiment).get("is_live"))

    def run_pid_status(self, experiment: str) -> dict:
        root = self.experiment_root(experiment)
        pid_path = root / "run_all.pid"
        pid_meta = parse_key_value_lines(safe_read_text(root / "run_all.pid.meta"))
        pid_text = safe_read_text(pid_path).strip()
        if not pid_text:
            fallback = find_live_pipeline_process(self.experiment_name(experiment))
            if fallback is not None:
                return {"pid": fallback["pid"], "running": True, "reason": f"{fallback['script']} process is alive"}
            return {"pid": None, "running": False, "reason": "no pid file"}
        try:
            pid = int(pid_text)
        except ValueError:
            return {"pid": pid_text, "running": False, "reason": "invalid pid file"}

        proc = Path("/proc") / str(pid)
        if not proc.exists():
            return {"pid": pid, "running": False, "reason": "pid not found"}

        expected_start_ticks = pid_meta.get("start_ticks")
        current_start_ticks = proc_start_ticks(proc)
        if expected_start_ticks and current_start_ticks and current_start_ticks != expected_start_ticks:
            return {"pid": pid, "running": False, "reason": "pid was reused"}

        cmdline = safe_read_bytes(proc / "cmdline").replace(b"\x00", b" ").decode("utf-8", errors="replace")
        known_scripts = ("run_all.sh", "run_out_lib_qvalue_sweep.sh", "run_existing_lib_searches.sh")
        if not any(script in cmdline for script in known_scripts):
            return {"pid": pid, "running": False, "reason": "pid belongs to another command"}
        script = next((script for script in known_scripts if script in cmdline), "pipeline")
        return {"pid": pid, "running": True, "reason": f"{script} pid is alive"}

    def load_progress_events(self, experiment: str) -> list[dict]:
        path = self.experiment_root(experiment) / "progress.txt"
        events = []
        for line in safe_read_text(path).splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            timestamp, step, state = parts[:3]
            details = parse_progress_details(parts[3] if len(parts) > 3 else "")
            events.append(
                {
                    "timestamp": timestamp,
                    "timestamp_dt": parse_progress_timestamp(timestamp),
                    "step": step,
                    "step_label": progress_step_label(step, details),
                    "state": state,
                    "details": details,
                }
            )
        return events

    def organism_for_experiment(self, experiment: str) -> str:
        root = self.experiment_root(experiment)
        progress = root / "progress.txt"
        if progress.exists():
            match = re.search(r"(?:^|[;\t ])organism=([^;\s]+)", safe_read_text(progress))
            if match:
                return normalize_organism(match.group(1))

        text_parts = [experiment]
        text_parts.extend(path.name for path in root.glob("*.fasta"))
        text_parts.extend(path.name for path in root.glob("*.fa"))
        command_files = [
            root / "all_startcmds.txt",
            root / "01_predlib" / "startcmd.txt",
            root / "05_quant" / "baseline" / "startcmd.txt",
        ]
        command_files.extend(sorted(root.glob("quant_search/carafe/*/*/logs/startcmd.txt")))
        command_files.extend(sorted(root.glob("external_searches/*/*/logs/startcmd.txt")))
        for path in command_files:
            if path.exists():
                text_parts.append(safe_read_text(path)[:5000])
        return infer_organism_from_text(" ".join(text_parts))

    def experiment_type_for_experiment(self, experiment: str) -> str:
        root = self.experiment_root(experiment)
        metadata = parse_key_value_lines(safe_read_text(root / "experiment_type.txt"))
        experiment_type = metadata.get("experiment_type", "").strip()
        if experiment_type:
            return normalize_experiment_type(experiment_type)

        text_parts = [experiment]
        for path in experiment_audit_paths(root):
            text_parts.append(str(path.relative_to(root)))
            text_parts.append(safe_read_text(path)[:5000])
        text = "\n".join(text_parts).lower()
        if "quant_search/carafe/" in text:
            return "Carafe comparison"
        if "run_out_lib_qvalue_sweep.sh" in text or "--out-lib-qvalue" in text or "__outlibq_" in text:
            return "out-lib q-value sweep"
        return "standard"

    def _has_search_result(self, experiment_root: Path) -> bool:
        quant_dir = experiment_root / "05_quant"
        if any_search_report_dir(quant_dir):
            return True

        # Carafe comparison searches intentionally live outside the run_all
        # numbered folders: quant_search/carafe/<profile>/<job>/results.
        if any_search_report_dir(experiment_root / "quant_search" / "carafe"):
            return True

        if any_external_search_report_dir(experiment_root / "external_searches"):
            return True

        # Make completed Step 3 learning-rate curves available before quantification.
        tune_root = experiment_root / "03_tune_models"
        return any(tune_root.glob("*/loss_plots/ALL_LR_LOSS.csv"))

    def list_searches(self, experiment: str) -> list[str]:
        root = self.experiment_root(experiment)
        searches = list_search_report_dirs(root / "05_quant")
        carafe_root = root / "quant_search" / "carafe"
        for relative in list_search_report_dirs(carafe_root):
            # Carafe stores reports one level below the logical search in
            # results/. Keep that implementation detail out of the UI id.
            logical_relative = relative.removesuffix("/results")
            search_id = f"carafe/{logical_relative}"
            if search_id not in searches:
                searches.append(search_id)
        for relative in list_external_search_report_dirs(root / "external_searches"):
            search_id = f"external/{relative}"
            if search_id not in searches:
                searches.append(search_id)
        return searches

    def _select_searches(self, experiment: str, searches: list[str] | None) -> list[str]:
        available = self.list_searches(experiment)
        if searches is None:
            return available
        requested = set(searches)
        selected = [search for search in available if search in requested]
        return selected or available

    def search_report_dir(self, experiment: str, search: str) -> Path:
        """Resolve standard and non-standard search layouts."""
        root = self.experiment_root(experiment)
        if search.startswith("carafe/"):
            search_dir = root / "quant_search" / "carafe" / search.removeprefix("carafe/")
            if (search_dir / "results").is_dir():
                return search_dir / "results"
            return search_dir
        if search.startswith("external/"):
            return root / "external_searches" / search.removeprefix("external/") / "05_quant"
        return root / "05_quant" / search

    def raw_report_path(self, experiment: str, search: str) -> Path:
        search_dir = self.search_report_dir(experiment, search)
        if search.startswith("carafe/"):
            for filename in ("final_report.parquet", "final_report-first-pass.parquet"):
                candidate = search_dir / filename
                if candidate.exists() and candidate.is_file():
                    return candidate
        path = preferred_report_path(search_dir)
        # A restricted-layer search is one DIA-NN run rooted at
        # ``tuned_restricted_layers/<profile>/lr_<value>``.  Evaluation
        # helpers may write report-shaped files in child directories.  Keep
        # those helpers from becoming additional searches, while retaining a
        # usable fallback for partially materialised runs that only have a
        # nested report so far.
        if path is None and restricted_layers_run_root(search) is not None:
            path = preferred_report_path_recursive(search_dir)
        if path is None:
            return search_dir / "final_report.parquet"
        return path

    def adapted_path(self, experiment: str, search: str, *, include_entrapment: bool = False) -> Path:
        first_pass = self.raw_report_path(experiment, search).name == "final_report-first-pass.parquet"
        if include_entrapment:
            stem = "adapted_first_pass_entrapment_primary" if first_pass else "adapted_entrapment_primary"
        else:
            stem = "adapted_first_pass" if first_pass else "adapted"
        filename = f"{stem}_{SERVER_CACHE_QVALUE_TAG}.parquet"
        return self.experiment_root(experiment) / "ma_plots" / "data" / search_name_for_cache(search) / filename

    def termini_path(self, experiment: str, search: str, *, include_entrapment: bool = False) -> Path:
        first_pass = self.raw_report_path(experiment, search).name == "final_report-first-pass.parquet"
        if include_entrapment:
            stem = "termini_first_pass_entrapment_primary" if first_pass else "termini_entrapment_primary"
        else:
            stem = "termini_first_pass" if first_pass else "termini"
        filename = f"{stem}_{SERVER_CACHE_QVALUE_TAG}.parquet"
        return self.experiment_root(experiment) / "ma_plots" / "data" / search_name_for_cache(search) / filename

    def upset_cache_dir(self, experiment: str) -> Path:
        return self.experiment_root(experiment) / "ma_plots" / "upset" / UPSET_CACHE_VERSION

    def triplemix_metrics_path(self, experiment: str, search: str) -> Path:
        return (
            self.experiment_root(experiment)
            / "ma_plots"
            / "triplemix"
            / TRIPLEMIX_CACHE_VERSION
            / search_name_for_cache(search)
            / "metrics.parquet"
        )

    def triplemix_report_path(self, experiment: str, search: str) -> Path:
        """Use the final Step-5 report like the QMD; fall back for one-pass searches."""
        search_dir = self.search_report_dir(experiment, search)
        return preferred_triplemix_report_path(search_dir) or self.raw_report_path(experiment, search)

    def upset_items_path(
        self,
        experiment: str,
        item_level: str,
        evidence_filter: str | list[str] | tuple[str, ...] | None,
        *,
        include_entrapment: bool = False,
        qvalue_threshold: float | None = None,
        searches: list[str] | None = None,
    ) -> Path:
        level = normalize_upset_item_level(item_level)
        evidence_key = evidence_filters_cache_key(evidence_filter)
        entrapment_key = "targets_entrapment" if include_entrapment else "targets"
        qvalue_key = qvalue_cache_key(qvalue_threshold)
        search_key = ""
        if searches is not None:
            digest = hashlib.sha1("\n".join(sorted(searches)).encode("utf-8")).hexdigest()[:10]
            search_key = f"_s{digest}"
        filename = f"upset_{level}_{evidence_key}_{entrapment_key}_{qvalue_key}{search_key}.parquet"
        return self.upset_cache_dir(experiment) / filename

    def upset_evidence_rows_path(
        self,
        experiment: str,
        item_level: str,
        evidence_filter: str | list[str] | tuple[str, ...] | None,
        *,
        include_entrapment: bool = False,
        qvalue_threshold: float | None = None,
    ) -> Path:
        level = normalize_upset_item_level(item_level)
        evidence_key = evidence_filters_cache_key(evidence_filter)
        entrapment_key = "targets_entrapment" if include_entrapment else "targets"
        qvalue_key = qvalue_cache_key(qvalue_threshold)
        filename = f"upset_evidence_{level}_{evidence_key}_{entrapment_key}_{qvalue_key}.parquet"
        return self.upset_cache_dir(experiment) / filename

    def load_raw(self, experiment: str, search: str, *, qvalue_threshold: float | None = None) -> pl.DataFrame:
        return self.load_raw_with_entrapment(experiment, search, include_entrapment=False, qvalue_threshold=qvalue_threshold)

    def load_raw_with_entrapment(
        self,
        experiment: str,
        search: str,
        *,
        include_entrapment: bool,
        include_diann_decoys: bool = False,
        qvalue_threshold: float | None = None,
    ) -> pl.DataFrame:
        df = cached_parquet(str(self.raw_report_path(experiment, search)))
        df = filter_by_qvalue(df, qvalue_threshold)
        if include_entrapment and not self.has_entrapment_hits(experiment):
            include_entrapment = False
        if include_entrapment:
            annotated = add_entrapment_columns(df)
            return annotated if include_diann_decoys else exclude_diann_decoys(annotated)
        return add_entrapment_columns(df) if include_diann_decoys else exclude_entrapment(df)

    def load_adapted(self, experiment: str, search: str, *, include_entrapment: bool = False) -> pl.DataFrame | None:
        self.ensure_maplot_cache(experiment, include_entrapment=include_entrapment, searches=[search])
        path = self.adapted_path(experiment, search, include_entrapment=include_entrapment)
        if not include_entrapment:
            return cached_parquet(str(path)) if path.exists() else None
        base_path = self.adapted_path(experiment, search)
        frames = [cached_parquet(str(candidate)) for candidate in (base_path, path) if candidate.exists()]
        return pl.concat(frames, how="diagonal_relaxed") if frames else None

    def load_termini(self, experiment: str, search: str, *, include_entrapment: bool = False) -> pl.DataFrame | None:
        self.ensure_maplot_cache(experiment, include_entrapment=include_entrapment, searches=[search])
        path = self.termini_path(experiment, search, include_entrapment=include_entrapment)
        if not include_entrapment:
            return cached_parquet(str(path)) if path.exists() else None
        base_path = self.termini_path(experiment, search)
        frames = [cached_parquet(str(candidate)) for candidate in (base_path, path) if candidate.exists()]
        return pl.concat(frames, how="diagonal_relaxed") if frames else None

    def load_all_raw(
        self,
        experiment: str,
        *,
        include_entrapment: bool = False,
        include_diann_decoys: bool = False,
        qvalue_threshold: float | None = None,
        searches: list[str] | None = None,
    ) -> dict[str, pl.DataFrame]:
        selected = self._select_searches(experiment, searches)
        return {
            search: self.load_raw_with_entrapment(
                experiment,
                search,
                include_entrapment=include_entrapment,
                include_diann_decoys=include_diann_decoys,
                qvalue_threshold=qvalue_threshold,
            )
            for search in selected
        }

    def build_upset_items_on_the_fly(
        self,
        experiment: str,
        item_level: str | None,
        evidence_filter: str | list[str] | tuple[str, ...] | None,
        *,
        include_entrapment: bool = False,
        qvalue_threshold: float | None = None,
        searches: list[str] | None = None,
    ) -> tuple[pl.DataFrame, list[str]]:
        searches = self._select_searches(experiment, searches)
        raw_tables = self.load_all_raw(
            experiment,
            include_entrapment=include_entrapment,
            qvalue_threshold=qvalue_threshold,
            searches=searches,
        )
        adapted_tables = self.load_all_adapted(experiment, include_entrapment=include_entrapment, searches=searches)
        termini_tables = self.load_all_termini(experiment, include_entrapment=include_entrapment, searches=searches)
        evidence_tables = evidence_tables_for_upset_cache(
            searches=searches,
            raw_tables=raw_tables,
            adapted_tables=adapted_tables,
            termini_tables=termini_tables,
            include_entrapment=include_entrapment,
        )
        return prepare_upset_items(
            evidence_tables,
            searches,
            normalize_upset_item_level(item_level),
            normalize_evidence_filters(evidence_filter),
            include_entrapment=include_entrapment,
        )

    def ensure_upset_cache(
        self,
        experiment: str,
        evidence_filter: str | list[str] | tuple[str, ...] | None,
        *,
        include_entrapment: bool = False,
        qvalue_threshold: float | None = None,
        item_levels: tuple[str, ...] | None = None,
        searches: list[str] | None = None,
    ) -> None:
        include_entrapment = include_entrapment and self.has_entrapment_hits(experiment)
        levels = item_levels or tuple(UPSET_LEVEL_TO_ID_PARTS)
        searches = self._select_searches(experiment, searches)
        missing_levels = [
            normalize_upset_item_level(level)
            for level in levels
            if self.upset_items_cache_is_stale(
                experiment,
                level,
                evidence_filter,
                searches,
                include_entrapment=include_entrapment,
                qvalue_threshold=qvalue_threshold,
            )
        ]
        if not missing_levels:
            return

        raw_tables = self.load_all_raw(
            experiment,
            include_entrapment=include_entrapment,
            qvalue_threshold=qvalue_threshold,
            searches=searches,
        )
        adapted_tables = self.load_all_adapted(experiment, include_entrapment=include_entrapment, searches=searches)
        termini_tables = self.load_all_termini(experiment, include_entrapment=include_entrapment, searches=searches)
        evidence_tables = evidence_tables_for_upset_cache(
            searches=searches,
            raw_tables=raw_tables,
            adapted_tables=adapted_tables,
            termini_tables=termini_tables,
            include_entrapment=include_entrapment,
        )
        normalized_filter = normalize_evidence_filters(evidence_filter)
        out_dir = self.upset_cache_dir(experiment)
        out_dir.mkdir(parents=True, exist_ok=True)
        for level in missing_levels:
            items, _set_cols = prepare_upset_items(
                evidence_tables,
                searches,
                level,
                normalized_filter,
                include_entrapment=include_entrapment,
            )
            items.write_parquet(
                self.upset_items_path(
                    experiment,
                    level,
                    normalized_filter,
                    include_entrapment=include_entrapment,
                    qvalue_threshold=qvalue_threshold,
                    searches=searches,
                )
            )
        cached_parquet.cache_clear()

    def upset_items_cache_is_stale(
        self,
        experiment: str,
        item_level: str,
        evidence_filter: str | list[str] | tuple[str, ...] | None,
        searches: list[str],
        *,
        include_entrapment: bool = False,
        qvalue_threshold: float | None = None,
    ) -> bool:
        path = self.upset_items_path(
            experiment,
            item_level,
            evidence_filter,
            include_entrapment=include_entrapment,
            qvalue_threshold=qvalue_threshold,
            searches=searches,
        )
        if not path.exists():
            return True
        try:
            columns = set(pl.read_parquet_schema(path))
        except Exception:
            return True
        return any(search not in columns for search in searches)

    def load_or_build_upset_items(
        self,
        experiment: str,
        item_level: str | None,
        evidence_filter: str | list[str] | tuple[str, ...] | None,
        *,
        include_entrapment: bool = False,
        qvalue_threshold: float | None = None,
        searches: list[str] | None = None,
    ) -> tuple[pl.DataFrame, list[str]]:
        include_entrapment = include_entrapment and self.has_entrapment_hits(experiment)
        searches = self._select_searches(experiment, searches)
        level = normalize_upset_item_level(item_level)
        normalized_filter = normalize_evidence_filters(evidence_filter)
        self.ensure_upset_cache(
            experiment,
            normalized_filter,
            include_entrapment=include_entrapment,
            qvalue_threshold=qvalue_threshold,
            item_levels=(level,),
            searches=searches,
        )
        path = self.upset_items_path(
            experiment,
            level,
            normalized_filter,
            include_entrapment=include_entrapment,
            qvalue_threshold=qvalue_threshold,
            searches=searches,
        )
        if not path.exists():
            return self.build_upset_items_on_the_fly(
                experiment,
                level,
                normalized_filter,
                include_entrapment=include_entrapment,
                qvalue_threshold=qvalue_threshold,
                searches=searches,
            )
        items = cached_parquet(str(path))
        set_cols = [search for search in searches if search in items.columns]
        return items, set_cols

    def build_upset_evidence_rows_on_the_fly(
        self,
        experiment: str,
        item_level: str | None,
        evidence_filter: str | list[str] | tuple[str, ...] | None,
        *,
        include_entrapment: bool = False,
        qvalue_threshold: float | None = None,
    ) -> pl.DataFrame:
        searches = self.list_searches(experiment)
        raw_tables = self.load_all_raw(
            experiment,
            include_entrapment=include_entrapment,
            qvalue_threshold=qvalue_threshold,
        )
        adapted_tables = self.load_all_adapted(experiment, include_entrapment=include_entrapment)
        termini_tables = self.load_all_termini(experiment, include_entrapment=include_entrapment)
        evidence_tables = evidence_tables_for_upset_cache(
            searches=searches,
            raw_tables=raw_tables,
            adapted_tables=adapted_tables,
            termini_tables=termini_tables,
            include_entrapment=include_entrapment,
        )
        return prepare_upset_evidence_rows(
            evidence_tables,
            searches,
            normalize_upset_item_level(item_level),
            normalize_evidence_filters(evidence_filter),
            include_entrapment=include_entrapment,
        )

    def load_or_build_upset_evidence_rows(
        self,
        experiment: str,
        item_level: str | None,
        evidence_filter: str | list[str] | tuple[str, ...] | None,
        *,
        include_entrapment: bool = False,
        qvalue_threshold: float | None = None,
    ) -> pl.DataFrame:
        include_entrapment = include_entrapment and self.has_entrapment_hits(experiment)
        level = normalize_upset_item_level(item_level)
        normalized_filter = normalize_evidence_filters(evidence_filter)
        path = self.upset_evidence_rows_path(
            experiment,
            level,
            normalized_filter,
            include_entrapment=include_entrapment,
            qvalue_threshold=qvalue_threshold,
        )
        if path.exists() and not self.upset_evidence_cache_is_stale(path, self.list_searches(experiment)):
            return cached_parquet(str(path))

        evidence = self.build_upset_evidence_rows_on_the_fly(
            experiment,
            level,
            normalized_filter,
            include_entrapment=include_entrapment,
            qvalue_threshold=qvalue_threshold,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_parquet(path)
        cached_parquet.cache_clear()
        return cached_parquet(str(path))

    def upset_evidence_cache_is_stale(self, path: Path, searches: list[str]) -> bool:
        try:
            schema = pl.read_parquet_schema(path)
            if "search" not in schema:
                return True
            cached_searches = set(pl.scan_parquet(path).select("search").unique().collect().get_column("search").to_list())
        except Exception:
            return True
        return any(search not in cached_searches for search in searches)

    @lru_cache(maxsize=128)
    def max_reported_qvalue(self, experiment: str) -> float | None:
        values = []
        for search in self.list_searches(experiment):
            path = self.raw_report_path(experiment, search)
            try:
                schema = pl.read_parquet_schema(path)
            except Exception:
                continue
            if "Q.Value" not in schema:
                continue
            value = (
                pl.scan_parquet(path)
                .select(pl.col("Q.Value").cast(pl.Float64, strict=False).max())
                .collect()
                .item()
            )
            if value is not None:
                values.append(float(value))
        return max(values) if values else None

    def ensure_maplot_cache(
        self,
        experiment: str,
        *,
        include_entrapment: bool = False,
        searches: list[str] | None = None,
    ) -> None:
        include_entrapment = include_entrapment and self.has_entrapment_hits(experiment)
        if include_entrapment:
            self.ensure_maplot_cache(experiment, include_entrapment=False, searches=searches)
        searches = self._select_searches(experiment, searches)
        if not searches:
            return
        organism = annotator_organism_name(self.organism_for_experiment(experiment))

        def termini_cache_is_stale(path: Path) -> bool:
            if not path.exists():
                return True
            if organism != "triplemix":
                return False
            try:
                return "protein_organism" not in pl.read_parquet_schema(path)
            except Exception:
                return True

        missing = [
            search
            for search in searches
            if not self.adapted_path(experiment, search, include_entrapment=include_entrapment).exists()
            or termini_cache_is_stale(self.termini_path(experiment, search, include_entrapment=include_entrapment))
        ]
        if not missing:
            return
        fasta_path = self.fasta_path_for_experiment(experiment)
        if fasta_path is None:
            return
        adapter = DiannAdapter(
            proteotypic=not include_entrapment,
            exclude_entrapment=not include_entrapment,
            qvalue_threshold=SERVER_CACHE_QVALUE_THRESHOLD,
        )
        annotator = TerminiAnnotator(organism=organism, sense="C", specificity="K|R", pssm_prediction=False)
        for search in missing:
            adapted_path = self.adapted_path(experiment, search, include_entrapment=include_entrapment)
            termini_path = self.termini_path(experiment, search, include_entrapment=include_entrapment)
            rebuild_termini = termini_cache_is_stale(termini_path)
            out_dir = adapted_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            raw = self.load_raw_with_entrapment(experiment, search, include_entrapment=include_entrapment)
            if include_entrapment and "is_entrapment" in raw.columns:
                raw = raw.filter(pl.col("is_entrapment").cast(pl.Boolean, strict=False).fill_null(False))
            adapted = cached_parquet(str(adapted_path)) if adapted_path.exists() else adapter.adapt(raw)
            if not adapted_path.exists():
                adapted.write_parquet(adapted_path)
            if rebuild_termini:
                termini = annotator.annotate(adapted, fasta_path)
                termini.write_parquet(termini_path)
        cached_parquet.cache_clear()

    def rebuild_maplot_cache(self, experiment: str, log=None) -> None:
        def emit(message: str) -> None:
            if log is not None:
                log(message)

        searches = self.list_searches(experiment)
        if not searches:
            emit(f"No DIA-NN final_report.parquet files found for {experiment}.")
            return

        fasta_path = self.fasta_path_for_experiment(experiment)
        if fasta_path is None:
            emit("No FASTA file found. Cannot create termini tables.")
            return

        organism = annotator_organism_name(self.organism_for_experiment(experiment))
        emit(f"Dataset: {experiment}")
        emit(f"Searches: {len(searches)}")
        emit(f"FASTA: {fasta_path}")
        emit(f"Organism annotation: {organism}")

        cached_parquet.cache_clear()
        adapter = DiannAdapter(qvalue_threshold=SERVER_CACHE_QVALUE_THRESHOLD)
        annotator = TerminiAnnotator(organism=organism, sense="C", specificity="K|R", pssm_prediction=False)
        for index, search in enumerate(searches, start=1):
            out_dir = self.adapted_path(experiment, search).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            emit(f"[{index}/{len(searches)}] Reading raw report: {search}")
            raw = self.load_raw(experiment, search)
            emit(f"[{index}/{len(searches)}] Adapting DIA-NN table")
            adapted = adapter.adapt(raw)
            adapted.write_parquet(self.adapted_path(experiment, search))
            emit(f"[{index}/{len(searches)}] Wrote {self.adapted_path(experiment, search)}")
            emit(f"[{index}/{len(searches)}] Annotating termini")
            termini = annotator.annotate(adapted, fasta_path)
            termini.write_parquet(self.termini_path(experiment, search))
            emit(f"[{index}/{len(searches)}] Wrote {self.termini_path(experiment, search)}")

        cached_parquet.cache_clear()
        emit("Done. Maplot tables rebuilt.")

    def fasta_path_for_experiment(self, experiment: str) -> Path | None:
        root = self.experiment_root(experiment)
        for path in sorted(root.glob("*.fasta")) + sorted(root.glob("*.fa")):
            return path
        command_files = [root / "all_startcmds.txt", root / "01_predlib" / "startcmd.txt"]
        command_files.extend(sorted(root.glob("quant_search/carafe/*/*/logs/startcmd.txt")))
        command_files.extend(sorted(root.glob("external_searches/*/*/logs/startcmd.txt")))
        for command_file in command_files:
            for path in fasta_paths_from_start_commands(safe_read_text(command_file)):
                if path.exists():
                    return path
        return None

    def has_entrapment_hits(self, experiment: str) -> bool:
        return self.entrapment_run_info(experiment)["enabled"]

    def entrapment_run_info(self, experiment: str) -> dict:
        root = self.experiment_root(experiment)
        paths = tuple(entrapment_audit_paths(root))
        return self._entrapment_run_info_cached(experiment, file_signature(paths))

    @lru_cache(maxsize=128)
    def _entrapment_run_info_cached(self, experiment: str, _signature: tuple[tuple[str, int, int], ...]) -> dict:
        root = self.experiment_root(experiment)
        text = "\n".join(safe_read_text(path) for path in entrapment_audit_paths(root))
        fasta_paths = fasta_paths_from_start_commands(text)
        entrapment_fasta_paths = []
        for path in fasta_paths:
            if ".entrapment_" not in path.name.lower() or not path.name.lower().endswith("combined.fasta"):
                continue
            # Cluster start commands can retain the original /pfs path while
            # the server sees the copied FASTA beside the reports.
            local_copy = root / path.name
            entrapment_fasta_paths.append(path if path.exists() else local_copy)
        entrapment_fasta_paths = unique_paths(entrapment_fasta_paths)
        valid_paths = [path for path in entrapment_fasta_paths if fasta_has_entrapment_headers(path)]
        return {
            "enabled": bool(valid_paths),
            "fasta_paths": [str(path) for path in entrapment_fasta_paths],
            "valid_fasta_paths": [str(path) for path in valid_paths],
        }

    def entrapment_summary(self, experiment: str) -> pl.DataFrame:
        if not self.has_entrapment_hits(experiment):
            return pl.DataFrame()
        return entrapment_summary(
            self.load_all_raw(experiment, include_entrapment=True, include_diann_decoys=True),
            compare_diann_decoys=True,
        )

    def entrapment_specificity_tables(self, experiment: str, specificity_group: str) -> dict[str, pl.DataFrame]:
        if not self.has_entrapment_hits(experiment):
            return {}
        return self._entrapment_specificity_tables_cached(
            experiment,
            specificity_group,
            self.entrapment_cache_signature(experiment),
        )

    @lru_cache(maxsize=64)
    def _entrapment_specificity_tables_cached(
        self,
        experiment: str,
        specificity_group: str,
        _signature: tuple[tuple[str, int, int], ...],
    ) -> dict[str, pl.DataFrame]:
        tables: dict[str, pl.DataFrame] = {}
        for search, table in self.entrapment_evidence_tables(experiment).items():
            if "specificity" not in table.columns:
                continue
            specificity = pl.col("specificity").cast(pl.Utf8)
            if specificity_group == "tryptic":
                table = table.filter(specificity == "specific")
            elif specificity_group == "semitryptic":
                table = table.filter(specificity.is_in(["semi_Nterm", "semi_Cterm"]))
            if not table.is_empty():
                tables[search] = table
        return tables

    def entrapment_evidence_tables(self, experiment: str) -> dict[str, pl.DataFrame]:
        if not self.has_entrapment_hits(experiment):
            return {}
        return self._entrapment_evidence_tables_cached(
            experiment,
            self.entrapment_cache_signature(experiment),
        )

    @lru_cache(maxsize=64)
    def _entrapment_evidence_tables_cached(
        self,
        experiment: str,
        _signature: tuple[tuple[str, int, int], ...],
    ) -> dict[str, pl.DataFrame]:
        raw_tables = self.load_all_raw(experiment, include_entrapment=True, include_diann_decoys=True)
        termini_tables = self.load_all_termini(experiment, include_entrapment=True)
        tables: dict[str, pl.DataFrame] = {}
        for search, raw in raw_tables.items():
            termini = termini_tables.get(search)
            table = raw_with_termini_evidence(raw, termini, include_entrapment=True) if termini is not None else raw
            if not table.is_empty():
                tables[search] = table
        return tables

    def entrapment_modification_tables(self, experiment: str, modification_group: str) -> dict[str, pl.DataFrame]:
        if not self.has_entrapment_hits(experiment):
            return {}
        return self._entrapment_modification_tables_cached(
            experiment,
            modification_group,
            self.entrapment_cache_signature(experiment),
        )

    def entrapment_tables_with_modification_groups(self, experiment: str) -> dict[str, pl.DataFrame]:
        if not self.has_entrapment_hits(experiment):
            return {}
        return self._entrapment_tables_with_modification_groups_cached(
            experiment,
            self.entrapment_cache_signature(experiment),
        )

    @lru_cache(maxsize=64)
    def _entrapment_tables_with_modification_groups_cached(
        self,
        experiment: str,
        _signature: tuple[tuple[str, int, int], ...],
    ) -> dict[str, pl.DataFrame]:
        tables: dict[str, pl.DataFrame] = {}
        for search, table in self.entrapment_evidence_tables(experiment).items():
            if table.is_empty():
                continue
            status_exprs = []
            if "nterm_status" in table.columns:
                status_exprs.append(pl.col("nterm_status"))
            if "nterm_modif" in table.columns:
                status_exprs.append(pl.col("nterm_modif"))
            if not status_exprs:
                continue
            tables[search] = table.with_columns(
                pl.coalesce(status_exprs)
                .map_elements(normalize_nterm_status, return_dtype=pl.Utf8)
                .alias("_nterm_mod_group")
            )
        return tables

    @lru_cache(maxsize=128)
    def _entrapment_modification_tables_cached(
        self,
        experiment: str,
        modification_group: str,
        _signature: tuple[tuple[str, int, int], ...],
    ) -> dict[str, pl.DataFrame]:
        tables: dict[str, pl.DataFrame] = {}
        for search, table in self.entrapment_tables_with_modification_groups(experiment).items():
            if table.is_empty():
                continue
            filtered = table.filter(pl.col("_nterm_mod_group") == modification_group).drop("_nterm_mod_group")
            if not filtered.is_empty():
                tables[search] = filtered
        return tables

    def entrapment_cache_signature(self, experiment: str) -> tuple[tuple[str, int, int], ...]:
        paths: list[Path] = []
        for search in self.list_searches(experiment):
            paths.append(self.raw_report_path(experiment, search))
            termini = self.termini_path(experiment, search, include_entrapment=True)
            if termini.exists():
                paths.append(termini)
        return file_signature(paths)

    def entrapment_summary_by_specificity(
        self,
        experiment: str,
        specificity_group: str,
        *,
        levels: tuple[str, ...] | None = None,
        qvalue_threshold: float | None = None,
    ) -> pl.DataFrame:
        normalized_levels = tuple(levels or ("precursor", "peptidoform", "peptide", "protein_group"))
        return self._entrapment_summary_by_specificity_cached(
            experiment,
            specificity_group,
            normalized_levels,
            qvalue_threshold,
            self.entrapment_cache_signature(experiment),
        )

    @lru_cache(maxsize=128)
    def _entrapment_summary_by_specificity_cached(
        self,
        experiment: str,
        specificity_group: str,
        levels: tuple[str, ...],
        qvalue_threshold: float | None,
        _signature: tuple[tuple[str, int, int], ...],
    ) -> pl.DataFrame:
        return entrapment_summary(
            self.entrapment_specificity_tables(experiment, specificity_group),
            levels=levels,
            qvalue_threshold=qvalue_threshold,
            compare_diann_decoys=True,
        )

    def entrapment_qvalue_curve_by_specificity(self, experiment: str, specificity_group: str, level: str = "peptide") -> pl.DataFrame:
        return self._entrapment_qvalue_curve_by_specificity_cached(
            experiment,
            specificity_group,
            level,
            self.entrapment_cache_signature(experiment),
        )

    @lru_cache(maxsize=128)
    def _entrapment_qvalue_curve_by_specificity_cached(
        self,
        experiment: str,
        specificity_group: str,
        level: str,
        _signature: tuple[tuple[str, int, int], ...],
    ) -> pl.DataFrame:
        return entrapment_qvalue_curve(
            self.entrapment_specificity_tables(experiment, specificity_group),
            level=level,
            compare_diann_decoys=True,
        )

    def entrapment_summary_by_modification(
        self,
        experiment: str,
        modification_group: str,
        *,
        levels: tuple[str, ...] | None = None,
        qvalue_threshold: float | None = None,
    ) -> pl.DataFrame:
        normalized_levels = tuple(levels or ("nterm_peptide", "nterm_site"))
        return self._entrapment_summary_by_modification_cached(
            experiment,
            modification_group,
            normalized_levels,
            qvalue_threshold,
            self.entrapment_cache_signature(experiment),
        )

    @lru_cache(maxsize=128)
    def _entrapment_summary_by_modification_cached(
        self,
        experiment: str,
        modification_group: str,
        levels: tuple[str, ...],
        qvalue_threshold: float | None,
        _signature: tuple[tuple[str, int, int], ...],
    ) -> pl.DataFrame:
        return entrapment_summary(
            self.entrapment_modification_tables(experiment, modification_group),
            levels=levels,
            qvalue_threshold=qvalue_threshold,
            compare_diann_decoys=True,
        )

    def entrapment_qvalue_curve_by_modification(self, experiment: str, modification_group: str, level: str = "nterm_peptide") -> pl.DataFrame:
        return self._entrapment_qvalue_curve_by_modification_cached(
            experiment,
            modification_group,
            level,
            self.entrapment_cache_signature(experiment),
        )

    @lru_cache(maxsize=128)
    def _entrapment_qvalue_curve_by_modification_cached(
        self,
        experiment: str,
        modification_group: str,
        level: str,
        _signature: tuple[tuple[str, int, int], ...],
    ) -> pl.DataFrame:
        return entrapment_qvalue_curve(
            self.entrapment_modification_tables(experiment, modification_group),
            level=level,
            compare_diann_decoys=True,
        )

    def entrapment_summary_all_evidence(
        self,
        experiment: str,
        *,
        levels: tuple[str, ...] | None = None,
        qvalue_threshold: float | None = None,
    ) -> pl.DataFrame:
        normalized_levels = tuple(levels or ("nterm_precursor", "nterm_protein"))
        return self._entrapment_summary_all_evidence_cached(
            experiment,
            normalized_levels,
            qvalue_threshold,
            self.entrapment_cache_signature(experiment),
        )

    @lru_cache(maxsize=128)
    def _entrapment_summary_all_evidence_cached(
        self,
        experiment: str,
        levels: tuple[str, ...],
        qvalue_threshold: float | None,
        _signature: tuple[tuple[str, int, int], ...],
    ) -> pl.DataFrame:
        return entrapment_summary(
            self.entrapment_evidence_tables(experiment),
            levels=levels,
            qvalue_threshold=qvalue_threshold,
            compare_diann_decoys=True,
        )

    def entrapment_qvalue_curve_all_evidence(self, experiment: str, level: str = "nterm_precursor") -> pl.DataFrame:
        return self._entrapment_qvalue_curve_all_evidence_cached(
            experiment,
            level,
            self.entrapment_cache_signature(experiment),
        )

    @lru_cache(maxsize=128)
    def _entrapment_qvalue_curve_all_evidence_cached(
        self,
        experiment: str,
        level: str,
        _signature: tuple[tuple[str, int, int], ...],
    ) -> pl.DataFrame:
        return entrapment_qvalue_curve(
            self.entrapment_evidence_tables(experiment),
            level=level,
            compare_diann_decoys=True,
        )

    def entrapment_qvalue_curve(self, experiment: str, level: str = "peptide") -> pl.DataFrame:
        if not self.has_entrapment_hits(experiment):
            return pl.DataFrame()
        return entrapment_qvalue_curve(
            self.load_all_raw(experiment, include_entrapment=True, include_diann_decoys=True),
            level=level,
            compare_diann_decoys=True,
        )

    def entrapment_qvalue_distribution(
        self,
        experiment: str,
        level: str = "peptide",
        *,
        max_qvalue: float | None = None,
        bin_width: float = 0.01,
    ) -> pl.DataFrame:
        if not self.has_entrapment_hits(experiment):
            return pl.DataFrame()
        return entrapment_qvalue_distribution(
            self.load_all_raw(experiment, include_entrapment=True, include_diann_decoys=True),
            level=level,
            max_qvalue=max_qvalue,
            bin_width=bin_width,
            compare_diann_decoys=True,
        )

    def entrapment_threshold_metrics(self, experiment: str, level: str = "peptide", qvalue_threshold: float = 0.01) -> pl.DataFrame:
        if not self.has_entrapment_hits(experiment):
            return pl.DataFrame()
        return entrapment_threshold_metrics(
            self.load_all_raw(experiment, include_entrapment=True, include_diann_decoys=True),
            level=level,
            qvalue_threshold=qvalue_threshold,
            compare_diann_decoys=True,
        )

    def load_all_adapted(
        self,
        experiment: str,
        *,
        include_entrapment: bool = False,
        searches: list[str] | None = None,
    ) -> dict[str, pl.DataFrame]:
        out = {}
        for search in self._select_searches(experiment, searches):
            df = self.load_adapted(experiment, search, include_entrapment=include_entrapment)
            if df is not None:
                out[search] = df
        return out

    def load_all_termini(
        self,
        experiment: str,
        *,
        include_entrapment: bool = False,
        searches: list[str] | None = None,
    ) -> dict[str, pl.DataFrame]:
        out = {}
        for search in self._select_searches(experiment, searches):
            df = self.load_termini(experiment, search, include_entrapment=include_entrapment)
            if df is not None:
                out[search] = df
        return out

    def load_triplemix_metrics(self, experiment: str) -> dict[str, pl.DataFrame]:
        """Build QMD-compatible Triplemix metrics directly from final Step-5 reports."""
        if self.organism_for_experiment(experiment) != "Mixed":
            return {}

        # Local import avoids coupling the repository's generic import path to Plotly.
        from .triplemix import TRIPLEMIX_QVALUE_THRESHOLD, build_triplemix_metrics, sample_condition_map

        tables: dict[str, pl.DataFrame] = {}
        experiment_root = self.experiment_root(experiment)
        fasta_path = self.fasta_path_for_experiment(experiment)
        if fasta_path is None:
            return {}
        organism = annotator_organism_name(self.organism_for_experiment(experiment))
        adapter = DiannAdapter(
            proteotypic=False,
            summarization="SUM",
            qvalue_threshold=TRIPLEMIX_QVALUE_THRESHOLD,
            exclude_entrapment=True,
        )
        annotator = TerminiAnnotator(organism=organism, sense="C", specificity="K|R", pssm_prediction=False)
        for search in self.list_searches(experiment):
            report_path = self.triplemix_report_path(experiment, search)
            if not report_path.exists():
                continue
            cache_path = self.triplemix_metrics_path(experiment, search)
            annotation_paths = [
                experiment_root / filename
                for filename in ("triplemix_sample_annotation.tsv", "sample_metadata_annotation.tsv")
                if (experiment_root / filename).exists()
            ]
            newest_source_mtime = max(path.stat().st_mtime_ns for path in [report_path, fasta_path, *annotation_paths])
            if triplemix_metrics_cache_is_current(
                cache_path,
                newest_source_mtime,
                report_path.name,
                TRIPLEMIX_QVALUE_THRESHOLD,
            ):
                tables[search] = cached_parquet(str(cache_path))
                continue

            raw = cached_parquet(str(report_path))
            adapted = adapter.adapt(raw)
            if adapted.is_empty():
                continue
            termini = annotator.annotate(adapted, fasta_path)
            sample_names = [column for column in adapted.columns if column not in ID_COLS]
            conditions = sample_condition_map(sample_names, experiment_root)
            metrics = build_triplemix_metrics(termini, condition_map=conditions)

            # Keep a genuinely pre-normalization diagnostic alongside both
            # method-specific views. Precursor.Quantity bypasses DIA-NN's
            # normalized quantity and the legacy MAD scaling.
            raw_quantity_column = "Precursor.Quantity" if "Precursor.Quantity" in raw.columns else "Precursor.Normalised"
            raw_adapter = DiannAdapter(
                proteotypic=False,
                summarization="SUM",
                qvalue_threshold=TRIPLEMIX_QVALUE_THRESHOLD,
                exclude_entrapment=True,
                quantity_column=raw_quantity_column,
                apply_mad_scaling=False,
            )
            raw_adapted = raw_adapter.adapt(raw)
            raw_termini = annotator.annotate(raw_adapted, fasta_path)
            raw_metrics = build_triplemix_metrics(raw_termini, condition_map=conditions).select(
                "nterm_modif_peptide",
                pl.col("abundance_A").alias("raw_abundance_A"),
                pl.col("abundance_B").alias("raw_abundance_B"),
                pl.col("fold_change").alias("raw_fold_change"),
                pl.col("mean_log10_abundance").alias("raw_mean_log10_abundance"),
                pl.col("complete_all_replicates").alias("raw_complete_all_replicates"),
            )
            metrics = metrics.join(raw_metrics, on="nterm_modif_peptide", how="left")
            metrics = metrics.with_columns(
                pl.lit(report_path.name).alias("source_report"),
                pl.lit(TRIPLEMIX_QVALUE_THRESHOLD).alias("source_qvalue_threshold"),
                pl.lit(False).alias("source_proteotypic_only"),
                pl.lit("SUM").alias("source_summarization"),
                pl.lit(raw_quantity_column).alias("source_raw_quantity"),
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            metrics.write_parquet(cache_path)
            tables[search] = metrics
        cached_parquet.cache_clear()
        return tables

    def list_ram_logs(self, experiment: str) -> list[dict]:
        root = self.experiment_root(experiment)
        logs = []
        for path in sorted(root.rglob("ram_*.txt")):
            step = ram_step_from_path(path)
            label = str(path.relative_to(root))
            logs.append(
                {
                    "path": str(path),
                    "label": label,
                    "short_label": ram_log_label(path),
                    "step": step,
                    "run": path.stem,
                }
            )
        return logs

    def load_ram_log(self, path: str) -> pd.DataFrame:
        return cached_ram_log(path)

def search_name_for_cache(search: str) -> str:
    return safe_cache_token(search.removeprefix("tuned/").replace("/", "__"))


def evidence_tables_for_upset_cache(
    *,
    searches: list[str],
    raw_tables: dict[str, pl.DataFrame],
    adapted_tables: dict[str, pl.DataFrame],
    termini_tables: dict[str, pl.DataFrame],
    include_entrapment: bool = False,
) -> dict[str, pl.DataFrame]:
    tables = {}
    for search in searches:
        if search in raw_tables and search in termini_tables:
            tables[search] = raw_with_termini_evidence(
                raw_tables[search],
                termini_tables[search],
                include_entrapment=include_entrapment,
            )
        elif search in termini_tables:
            tables[search] = termini_tables[search]
        elif search in adapted_tables:
            tables[search] = adapted_tables[search]
        elif search in raw_tables:
            tables[search] = raw_tables[search]
    return tables


def evidence_filters_cache_key(value: str | list[str] | tuple[str, ...] | None) -> str:
    return safe_cache_token("_".join(normalize_evidence_filters(value)))


def qvalue_cache_key(value: float | None) -> str:
    if value is None:
        return "qall"
    text = f"{float(value):g}".replace(".", "p").replace("-", "m")
    return safe_cache_token(f"q{text}")


def safe_cache_token(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return token.strip("_") or "none"


def preferred_report_path(search_dir: Path) -> Path | None:
    for filename in PREFERRED_REPORT_FILES:
        candidate = search_dir / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def preferred_report_path_recursive(search_dir: Path) -> Path | None:
    """Return one deterministic report below a restricted-layer run root."""
    direct = preferred_report_path(search_dir)
    if direct is not None:
        return direct
    candidates = [
        path
        for filename in PREFERRED_REPORT_FILES
        for path in search_dir.rglob(filename)
        if path.is_file() and ".interrupted" not in path.parts
    ]
    if not candidates:
        return None
    # Keep report preference (first-pass before final) and make the fallback
    # stable even if DIA-NN evaluation output is created in parallel.
    preference = {name: index for index, name in enumerate(PREFERRED_REPORT_FILES)}
    return min(candidates, key=lambda path: (preference[path.name], path.as_posix()))


def restricted_layers_run_root(search: str) -> str | None:
    """Canonical Step-5 id for a DIA-NN restricted-layer variant.

    The Step-5 runner stores a variant below
    ``tuned_restricted_layers/<profile>/lr_<learning-rate>/``.  Additional
    descendants (for example ``eval/arabidopsis_reference``) are analysis artefacts, not
    independent DIA-NN searches.
    """
    parts = tuple(part for part in str(search).strip("/").split("/") if part)
    if len(parts) < 3:
        return None
    variant, profile, lr_dir = parts[:3]
    is_restricted_layers = (
        variant.startswith(RESTRICTED_LAYERS_RUN_PREFIX)
        and "restricted" in variant.lower()
        and "layer" in variant.lower()
    )
    if not is_restricted_layers or not profile or not lr_dir.startswith("lr_"):
        return None
    return "/".join((variant, profile, lr_dir))


def canonical_search_id(search: str) -> str:
    """Collapse restricted-layer child artefacts into their parent run id."""
    return restricted_layers_run_root(search) or search


def preferred_triplemix_report_path(search_dir: Path) -> Path | None:
    for filename in TRIPLEMIX_REPORT_FILES:
        candidate = search_dir / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def triplemix_metrics_cache_is_current(
    cache_path: Path,
    newest_source_mtime_ns: int,
    source_report: str,
    qvalue_threshold: float,
) -> bool:
    required = {
        "source_report",
        "source_qvalue_threshold",
        "source_proteotypic_only",
        "source_summarization",
        "complete_all_replicates",
        "mean_log10_abundance",
        "arabidopsis_reference_fold_change_normalized",
        "arabidopsis_reference_mean_log10_abundance",
        "arabidopsis_reference_center",
        "raw_fold_change",
        "raw_mean_log10_abundance",
        "raw_complete_all_replicates",
        "source_raw_quantity",
        "excluded_cross_organism_termini",
    }
    if not cache_path.exists() or cache_path.stat().st_mtime_ns < newest_source_mtime_ns:
        return False
    try:
        if not required.issubset(pl.read_parquet_schema(cache_path)):
            return False
        metadata = (
            pl.scan_parquet(cache_path)
            .select("source_report", "source_qvalue_threshold", "source_proteotypic_only", "source_summarization")
            .head(1)
            .collect()
        )
    except (OSError, pl.exceptions.PolarsError):
        return False
    if metadata.is_empty():
        return False
    return (
        metadata.item(0, "source_report") == source_report
        and float(metadata.item(0, "source_qvalue_threshold")) == float(qvalue_threshold)
        and metadata.item(0, "source_proteotypic_only") is False
        and metadata.item(0, "source_summarization") == "SUM"
    )


def cached_parquet(path: str) -> pl.DataFrame:
    stat = Path(path).stat()
    return _cached_parquet(path, stat.st_mtime_ns, stat.st_size)


def filter_by_qvalue(df: pl.DataFrame, threshold: float | None) -> pl.DataFrame:
    if threshold is None or "Q.Value" not in df.columns:
        return df
    return df.filter(pl.col("Q.Value").cast(pl.Float64, strict=False) <= float(threshold))


@lru_cache(maxsize=128)
def _cached_parquet(path: str, _mtime_ns: int, _size: int) -> pl.DataFrame:
    return pl.read_parquet(path)


cached_parquet.cache_clear = _cached_parquet.cache_clear  # type: ignore[attr-defined]


@lru_cache(maxsize=128)
def cached_ram_log(path: str) -> pd.DataFrame:
    rows = []
    pattern = re.compile(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})\s+\|\s+"
        r"Pipeline:\s+(?P<pipeline>[-0-9.]+)\s+GB\s+\|\s+"
        r"SysUsed:\s+(?P<sysused>[-0-9.]+)\s+GB\s+\|\s+"
        r"Avail:\s+(?P<avail>[-0-9.]+)\s+GB\s+\|\s+"
        r"Swap:\s+(?P<swap>[-0-9.]+)\s+GB"
    )
    for line in Path(path).read_text(errors="ignore").splitlines():
        match = pattern.search(line.strip())
        if not match:
            continue
        rows.append(match.groups())
    data = pd.DataFrame(rows, columns=["timestamp", "pipeline_gb", "system_used_gb", "available_gb", "swap_gb"])
    if data.empty:
        return data
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y-%m-%d_%H:%M:%S", errors="coerce")
    for col in ("pipeline_gb", "system_used_gb", "available_gb", "swap_gb"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp")
    start = data["timestamp"].min()
    data["elapsed_min"] = (data["timestamp"] - start).dt.total_seconds() / 60
    return data


def sample_columns(df: pl.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in ID_COLS]


def any_search_report_dir(quant_dir: Path) -> bool:
    return bool(list_search_report_dirs(quant_dir))


def list_search_report_dirs(quant_dir: Path) -> list[str]:
    if not quant_dir.exists():
        return []

    searches = {
        canonical_search_id(str(path.parent.relative_to(quant_dir)))
        for filename in PREFERRED_REPORT_FILES
        for path in quant_dir.rglob(filename)
        if path.is_file() and ".interrupted" not in path.parts
    }

    def sort_key(search: str) -> tuple[int, str]:
        if search == "baseline":
            return (0, search)
        if search.endswith("/baseline"):
            return (1, search)
        if search.startswith("tuned/"):
            return (2, search)
        return (3, search)

    return sorted(searches, key=sort_key)


def any_external_search_report_dir(external_root: Path) -> bool:
    return bool(list_external_search_report_dirs(external_root))


def list_external_search_report_dirs(external_root: Path) -> list[str]:
    """List external/<model-id>/<search-id> runs with a completed Step-5 report."""
    if not external_root.exists():
        return []
    searches = []
    for model_dir in sorted(path for path in external_root.iterdir() if path.is_dir()):
        for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            if preferred_report_path(run_dir / "05_quant") is not None:
                searches.append(f"{model_dir.name}/{run_dir.name}")
    return searches


def entrapment_audit_paths(root: Path) -> list[Path]:
    candidates = [
        root / "all_startcmds.txt",
        root / "01_predlib" / "startcmd.txt",
        root / "05_quant" / "baseline" / "startcmd.txt",
    ]
    candidates.extend(sorted(root.glob("05_quant/tuned/*/startcmd.txt")))
    candidates.extend(sorted(root.glob("05_quant/*/*/startcmd.txt")))
    candidates.extend(sorted(root.glob("quant_search/carafe/*/*/logs/startcmd.txt")))
    candidates.extend(sorted(root.glob("external_searches/*/*/logs/startcmd.txt")))
    candidates.extend(sorted(root.glob("04_refined_lib*/**/startcmd*.txt")))
    seen = set()
    paths = []
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        paths.append(path)
    return paths


def experiment_audit_paths(root: Path) -> list[Path]:
    candidates = [
        root / "experiment_type.txt",
        root / "all_startcmds.txt",
        root / "01_predlib" / "startcmd.txt",
    ]
    candidates.extend(sorted(root.glob("05_quant/*/startcmd.txt")))
    candidates.extend(sorted(root.glob("05_quant/tuned/*/startcmd.txt")))
    candidates.extend(sorted(root.glob("05_quant/*/*/startcmd.txt")))
    candidates.extend(sorted(root.glob("quant_search/carafe/*/*/logs/startcmd.txt")))
    candidates.extend(sorted(root.glob("external_searches/*/*/logs/startcmd.txt")))
    seen = set()
    paths = []
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        paths.append(path)
    return paths


def fasta_paths_from_start_commands(text: str) -> list[Path]:
    paths = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            tokens = shlex.split(stripped)
        except ValueError:
            tokens = stripped.split()
        for idx, token in enumerate(tokens):
            if token in {"--fasta", "--fastaabs", "--fasta-abs"} and idx + 1 < len(tokens):
                paths.append(Path(tokens[idx + 1]))
            elif token.startswith("--fasta="):
                paths.append(Path(token.split("=", 1)[1]))
            elif token.startswith("--fastaabs="):
                paths.append(Path(token.split("=", 1)[1]))
            elif token.startswith("--fasta-abs="):
                paths.append(Path(token.split("=", 1)[1]))
    return paths


def unique_paths(paths) -> list[Path]:
    out = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def file_signature(paths: Iterable[Path]) -> tuple[tuple[str, int, int], ...]:
    signature = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            signature.append((str(path), -1, -1))
            continue
        signature.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def fasta_has_entrapment_headers(path: Path) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    return _cached_fasta_has_entrapment_headers(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=128)
def _cached_fasta_has_entrapment_headers(path: str, _mtime_ns: int, _size: int) -> bool:
    try:
        with Path(path).open("r", errors="ignore") as handle:
            for line in handle:
                if line.startswith(">ENT_"):
                    return True
    except OSError:
        return False
    return False


def ram_step_from_path(path: Path) -> str:
    text = str(path).lower()
    if "01_predlib" in text or "step1" in path.name:
        return "step1_predlib"
    if "02_tunelib" in text or "step2" in path.name:
        return "step2_tunelib"
    if "03_tune_models" in text or "step3" in path.name:
        return "step3_tune_models"
    if "04_refined" in text or "step4" in path.name:
        return "step4_refined_lib"
    if "05_quant" in text or "step5" in path.name:
        return "step5_quant"
    if "make_cal" in path.name:
        return "make_cal_lib"
    return "other"


def ram_log_label(path: Path) -> str:
    step = ram_step_from_path(path)
    step_label = {
        "step1_predlib": "Step 1",
        "step2_tunelib": "Step 2",
        "step3_tune_models": "Step 3",
        "step4_refined_lib": "Step 4",
        "step5_quant": "Step 5",
        "make_cal_lib": "Cal lib",
        "other": "Other",
    }.get(step, step)
    run = ram_run_label(path, step)
    return clean_ram_label(" · ".join(part for part in (step_label, run) if part))


def ram_run_label(path: Path, step: str) -> str:
    parent_parts = path.parts
    name = path.stem
    if step == "step1_predlib":
        return "predlib"
    if step == "step2_tunelib":
        return parent_parts[-2] if len(parent_parts) >= 2 else "tunelib"
    if step == "step3_tune_models":
        return parent_parts[-2] if len(parent_parts) >= 2 else "tune models"
    if step == "step4_refined_lib":
        model = "rt+fr" if "_rt_fr_" in name else "rt" if "_rt_" in name else "refined"
        profile = parent_parts[-4] if len(parent_parts) >= 4 else ""
        return " ".join(part for part in (profile, model) if part)
    if step == "step5_quant":
        if "baseline" in path.parts:
            return "baseline"
        qvalue_match = re.search(r"q_([0-9_]+)__outlibq_([0-9_]+)", str(path))
        if qvalue_match:
            qvalue = qvalue_match.group(1).replace("_", ".")
            outlib_qvalue = qvalue_match.group(2).replace("_", ".")
            return f"q {qvalue} / out-lib {outlib_qvalue}"
        model = "rt+fr" if "_rt_fr_" in name else "rt" if "_rt_" in name else ""
        profile_match = re.search(r"ram_step5_tuned_(.*?)_\d{8}_\d{6}$", name)
        profile = profile_match.group(1) if profile_match else "tuned"
        profile = re.sub(r"_lr_.*$", "", profile).replace("_rt_fr", "").replace("_rt", "").replace("_", " ")
        return " ".join(part for part in (profile, model) if part)
    if step == "make_cal_lib":
        match = re.search(r"ram_make_cal_(.*?)_\d{8}_\d{6}$", name)
        if match:
            return match.group(1).replace("_", " ")
        return "make cal"
    return clean_ram_label(name.replace("_", " "))


def clean_ram_label(label: str) -> str:
    label = re.sub(r"\s*[·-]\s*\d{8}\s+\d{6}\s*$", "", label)
    label = re.sub(r"[_\s-]+\d{8}[_\s-]*\d{6}\s*$", "", label)
    label = re.sub(r"\s+", " ", label.replace("_", " ")).strip()
    label = re.sub(r"\s+·\s+", " · ", label)
    return label


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return ""


def safe_read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


def loss_plot_title(filename: str, profile: str, kind: str, model: str, distill: int | None, lr_tag: str | None) -> str:
    model_label = model if model and model != "other" else "Loss"
    if kind == "combined":
        return f"{profile} - all LR {model_label} loss d{distill if distill is not None else 0}"
    if kind == "distillate":
        return f"{profile} - distillate d{distill} {model_label} loss"
    if lr_tag:
        return f"{profile} - LR {lr_tag.replace('p', '.')} {model_label} loss"
    return f"{profile} - {filename}"


def parse_key_value_lines(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def normalize_experiment_type(value: str) -> str:
    token = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "out_lib_qvalue_sweep": "out-lib q-value sweep",
        "out_lib_q_value_sweep": "out-lib q-value sweep",
        "outlib_qvalue_sweep": "out-lib q-value sweep",
        "qvalue_sweep": "out-lib q-value sweep",
        "standard": "standard",
        "run_all": "standard",
    }
    return aliases.get(token, value.strip().replace("_", " ") or "standard")


def find_live_pipeline_process(experiment: str) -> dict | None:
    proc_root = Path("/proc")
    scripts = ("run_all.sh", "run_out_lib_qvalue_sweep.sh", "run_existing_lib_searches.sh")
    for proc in proc_root.iterdir():
        if not proc.name.isdigit():
            continue
        cmdline = safe_read_bytes(proc / "cmdline")
        if not cmdline:
            continue
        text = cmdline.replace(b"\x00", b" ").decode("utf-8", errors="replace")
        if experiment not in text:
            continue
        script = next((candidate for candidate in scripts if candidate in text), None)
        if script is None:
            continue
        return {"pid": int(proc.name), "script": script}
    return None


def live_activity_is_stale(last_activity_ts: float | int | None) -> bool:
    if not last_activity_ts:
        return False
    return datetime.now(tz=timezone.utc).timestamp() - float(last_activity_ts) > LIVE_STALE_AFTER_SECONDS


def proc_start_ticks(proc: Path) -> str | None:
    stat = safe_read_text(proc / "stat")
    if not stat:
        return None
    parts = stat.rsplit(") ", 1)
    fields = parts[1].split() if len(parts) == 2 else stat.split()
    return fields[19] if len(fields) > 19 else None


def parse_progress_timestamp(value: str):
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        return None


def parse_progress_details(text: str) -> dict[str, str]:
    details = {}
    for chunk in text.split(";"):
        if "=" in chunk:
            key, value = chunk.split("=", 1)
            details[key.strip()] = value.strip()
    return details


def progress_step_label(step: str, details: dict[str, str]) -> str:
    base = {
        "step1_predlib.sh": "Step 1 predlib",
        "make_cal_lib.sh": "Calibration lib",
        "step2_tunelib.sh": "Step 2 tunelib",
        "step3_tune_models.sh": "Step 3 tune models",
        "step4_create_tuned_predlib.sh": "Step 4 refined lib",
        "step5_quant.sh": "Step 5 quant",
        "run_existing_lib_searches.sh": "Step 5 re-search",
        "maplot_report": "MA/QC report",
    }.get(step, step)
    if details.get("mode") == "out-lib-qvalue-sweep":
        bits = [
            details.get("mode"),
            f"q {details['qvalue']}" if details.get("qvalue") else "",
            f"out-lib {details['out_lib_qvalue']}" if details.get("out_lib_qvalue") else "",
        ]
    elif step == "run_existing_lib_searches.sh":
        bits = [details.get("group"), details.get("search"), details.get("mode")]
    else:
        bits = [details.get("profile"), details.get("models"), details.get("mode")]
    suffix = " · ".join(bit for bit in bits if bit)
    return f"{base} · {suffix}" if suffix else base


def current_progress_step(events: list[dict]) -> str | None:
    open_steps: dict[tuple[str, tuple[tuple[str, str], ...]], dict] = {}
    for event in events:
        key_details = tuple(sorted((key, value) for key, value in event["details"].items() if key not in {"elapsed"}))
        key = (event["step"], key_details)
        if event["state"] == "START":
            open_steps[key] = event
        elif event["state"] in {"OK", "FAIL"}:
            open_steps.pop(key, None)
    if not open_steps:
        return None
    latest = max(open_steps.values(), key=lambda event: event["timestamp_dt"] or datetime.min.replace(tzinfo=timezone.utc))
    return latest["step_label"]


def normalize_organism(value: str) -> str:
    cleaned = value.strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "arabidopsis": "Arabidopsis thaliana",
        "arabidopsis thaliana": "Arabidopsis thaliana",
        "human": "Homo sapiens",
        "homo sapiens": "Homo sapiens",
        "celegans": "Caenorhabditis elegans",
        "c elegans": "Caenorhabditis elegans",
        "caenorhabditis elegans": "Caenorhabditis elegans",
        "ecoli": "Escherichia coli",
        "escherichia coli": "Escherichia coli",
        "mixed": "Mixed",
        "triple mix": "Mixed",
        "triplemix": "Mixed",
    }
    return aliases.get(cleaned, value.strip() or "Unknown")


def annotator_organism_name(value: str) -> str:
    cleaned = normalize_organism(value)
    aliases = {
        "Homo sapiens": "human",
        "Arabidopsis thaliana": "arabidopsis",
        "Caenorhabditis elegans": "c_elegans",
        "Escherichia coli": "ecoli",
        # The server uses the display label "Mixed" for a triplemix dataset,
        # but TerminiAnnotator needs the concrete supported organism key.
        "Mixed": "triplemix",
    }
    return aliases.get(cleaned, value.strip().lower().replace(" ", "_") or "human")


def infer_organism_from_text(text: str) -> str:
    lower = text.lower()
    hits = set()
    if "homo_sapiens_arabidopsis_thaliana_escherichia_coli" in lower or "triplemix" in lower:
        return "Mixed"
    if "up000006548" in lower or "arabidopsis" in lower or "thaliana" in lower or "_ara" in lower or lower.startswith("ara"):
        hits.add("Arabidopsis thaliana")
    if "up000001940" in lower or "celegans" in lower or "c elegans" in lower or "caenorhabditis" in lower or lower.startswith("maya"):
        hits.add("Caenorhabditis elegans")
    if "uniprot_proteome_human" in lower or "uniprotkb_proteome_human" in lower or "homo sapiens" in lower or "homo_sapiens" in lower:
        hits.add("Homo sapiens")
    if "escherichia_coli" in lower or "e coli" in lower or "ecoli" in lower:
        hits.add("Escherichia coli")

    if len(hits) > 1:
        return "Mixed"
    if hits:
        return next(iter(hits))
    return "Unknown"
