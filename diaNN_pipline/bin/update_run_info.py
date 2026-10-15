#!/usr/bin/env python3
"""Build a lightweight, refreshable catalogue of DIA-NN pipeline runs.

The script reads run folders only and writes derived documentation/database
files to a separate output directory.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shlex
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(os.environ.get("DIANN_OUTROOT", str(Path.home() / "ma" / "diann_out")))
DEFAULT_OUT = DEFAULT_ROOT / "run_info"
FASTA_FALLBACK_DIRS = [
    Path(path)
    for path in os.environ.get(
        "DIANN_FASTA_FALLBACK_DIRS",
        os.pathsep.join(
            (
                str(Path.home() / "ma" / "fastas"),
                str(Path.home() / "media" / "hs_test_pipeline" / "test_fastas"),
            )
        ),
    ).split(os.pathsep)
    if path
]
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRS = [PIPELINE_ROOT / "datasets"]
MOD_CONFIG_DIR = PIPELINE_ROOT / "config" / "mods"

STEP_LABELS = {
    "step1_predlib.sh": "1 predlib",
    "make_calibration_lib": "calibration",
    "make_cal_lib.sh": "calibration",
    "step2_tunelib.sh": "2 tunelib searches",
    "step3_tune_models.sh": "3 tune models",
    "step4_create_tuned_predlib.sh": "4 refined library",
    "step5_quant.sh": "5 quant searches",
    "maplot_report": "MA/QC report",
}

ORGANISM_ALIASES = {
    "arabidopsis": ("Arabidopsis thaliana", "3702"),
    "ara": ("Arabidopsis thaliana", "3702"),
    "human": ("Homo sapiens", "9606"),
    "hela": ("Homo sapiens", "9606"),
    "celegans": ("Caenorhabditis elegans", "6239"),
    "cele": ("Caenorhabditis elegans", "6239"),
    "maya": ("Caenorhabditis elegans", "6239"),
}

UNIMOD_NAMES = {
    "UniMod:1": "Acetyl",
    "UniMod:4": "Carbamidomethyl",
    "UniMod:35": "Oxidation",
    "UniMod:36": "Dimethyl",
    "UniMod:510": "Dimethyl:2H(4)13C(2)",
}

SITE_LABELS = {
    "n": "peptide N-term",
    "K": "Lysine (K)",
    "C": "Cysteine (C)",
    "M": "Methionine (M)",
}

STATUS_BADGES = {
    "complete": "✅ complete",
    "failed": "❌ failed",
    "partial": "🟡 partial",
    "unknown": "⚪ unknown",
}


def read_text(path: Path, limit: int | None = None) -> str:
    try:
        if limit is None:
            return path.read_text(errors="replace")
        with path.open("r", errors="replace") as handle:
            return handle.read(limit)
    except OSError:
        return ""


def parse_simple_yaml_metadata(path: Path) -> dict[str, Any]:
    """Parse the small subset of dataset YAML we need without extra deps."""
    text = read_text(path)
    data: dict[str, Any] = {"dataset_file": str(path)}
    patterns = {
        "dataset_id": r"(?m)^\s*dataset_id:\s*[\"']?([^\"'\n#]+)",
        "organism": r"(?m)^\s*name:\s*[\"']?([^\"'\n#]+)",
        "taxonomy_id": r"(?m)^\s*taxonomy_id:\s*[\"']?([^\"'\n#]+)",
        "fasta": r"(?m)^\s*fasta:\s*[\"']?([^\"'\n#]+)",
        "fasta_swiss": r"(?m)^\s*fasta_swiss:\s*[\"']?([^\"'\n#]+)",
        "base_dir": r"(?m)^\s*base_dir:\s*[\"']?([^\"'\n#]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            data[key] = match.group(1).strip()
    for source in ("prehunter", "hunter"):
        block = re.search(rf"(?ms)^\s{{2}}{source}:\s*(.*?)(?=^\s{{2}}\w+:|\Z)", text)
        if block:
            section = block.group(1)
            date = re.search(r"(?m)^\s*date:\s*[\"']?([^\"'\n#]+)", section)
            instrument = re.search(r"(?m)^\s*instrument:\s*[\"']?([^\"'\n#]+)", section)
            train = re.findall(r"(?ms)^\s*train_files:\s*(.*?)(?=^\s*\w|^\s{2}\w|\Z)", section)
            test = re.findall(r"(?ms)^\s*test_files:\s*(.*?)(?=^\s*\w|^\s{2}\w|\Z)", section)
            data[f"{source}_date"] = date.group(1).strip() if date else None
            data[f"{source}_instrument"] = instrument.group(1).strip() if instrument else None
            data[f"{source}_train_count"] = count_yaml_list(train[0]) if train else None
            data[f"{source}_test_count"] = count_yaml_list(test[0]) if test else None
    return data


def count_yaml_list(text: str) -> int:
    return len(re.findall(r"(?m)^\s*-\s+", text))


def load_dataset_metadata() -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for directory in DATASET_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.yaml")):
            data = parse_simple_yaml_metadata(path)
            dataset_id = data.get("dataset_id") or path.stem
            metadata[dataset_id] = data
    return metadata


def shell_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def option_values(tokens: list[str], option: str) -> list[str]:
    values: list[str] = []
    for index, token in enumerate(tokens):
        if token == option and index + 1 < len(tokens):
            values.append(tokens[index + 1])
        elif token.startswith(option + "="):
            values.append(token.split("=", 1)[1])
    return values


def option_value(tokens: list[str], option: str) -> str | None:
    values = option_values(tokens, option)
    return values[-1] if values else None


def parse_set_values(tokens: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for index, token in enumerate(tokens):
        if token == "--set" and index + 1 < len(tokens) and "=" in tokens[index + 1]:
            key, value = tokens[index + 1].split("=", 1)
            values[key] = value.replace("\\,", ",")
    return values


def resolve_mod_config_path(value: str | None) -> Path | None:
    if not value:
        return None
    candidates = []
    raw = Path(value)
    candidates.append(raw)
    candidates.append(MOD_CONFIG_DIR / value)
    if raw.suffix != ".cfg":
        candidates.append(MOD_CONFIG_DIR / f"{value}.cfg")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_mod_config(value: str | None) -> dict[str, Any]:
    path = resolve_mod_config_path(value)
    if not path:
        return {"config": value, "path": None, "fixed": [], "variable": [], "note": "config file not found"}
    mods = {"config": path.name, "path": str(path), "fixed": [], "variable": [], "note": ""}
    current: str | None = None
    for raw_line in read_text(path).splitlines():
        line = raw_line.strip()
        if line.startswith("FIXED_MODS_LIST"):
            current = "fixed"
            continue
        if line.startswith("VAR_MODS_LIST"):
            current = "variable"
            continue
        if current and line.startswith(")"):
            current = None
            continue
        if not current:
            continue
        match = re.search(r'"([^"]+)"(?:\s*#\s*(.*))?', line)
        if not match:
            continue
        parts = [part.strip() for part in match.group(1).split(",")]
        if len(parts) < 3:
            continue
        unimod, mass, sites = parts[:3]
        mods[current].append(
            {
                "unimod": unimod,
                "name": UNIMOD_NAMES.get(unimod, unimod),
                "mass": mass,
                "sites": sites,
                "site_labels": describe_mod_sites(sites),
                "comment": (match.group(2) or "").strip(),
            }
        )
    return mods


def describe_mod_sites(sites: str) -> list[str]:
    labels = []
    for char in sites:
        labels.append(SITE_LABELS.get(char, char))
    return labels or [sites]


def command_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("bash ", "/", "diann-", "run_all.sh")):
            lines.append(line)
    return lines


def parse_run_invocation(all_startcmds: str) -> dict[str, Any]:
    for line in command_lines(all_startcmds):
        if "run_all.sh" not in line:
            continue
        tokens = shell_tokens(line)
        return {
            "expname": option_value(tokens, "--expname"),
            "dataset": option_value(tokens, "--ds"),
            "profile_request": option_value(tokens, "--profile"),
            "mods_config": option_value(tokens, "--mods"),
            "fasta_kind": option_value(tokens, "--fasta-kind"),
            "fasta_abs": option_value(tokens, "--fasta-abs"),
            "no_cal_lib": "--no-cal-lib" in tokens,
            "ram_log": "--ram-log" in tokens,
            "ram_log_steps": option_value(tokens, "--ram-log-steps"),
            "profile_parallel": "--profile-parallel" in tokens,
            "set_values": parse_set_values(tokens),
        }
    return {}


def parse_progress(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = []
    text = read_text(path)
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        timestamp, step, state = parts[:3]
        details = parse_detail_fields(parts[3] if len(parts) > 3 else "")
        events.append(
            {
                "timestamp": timestamp,
                "step": step,
                "label": STEP_LABELS.get(step, step),
                "state": state,
                "details": details,
            }
        )
    summary: dict[str, Any] = {
        "event_count": len(events),
        "started_at": events[0]["timestamp"] if events else None,
        "ended_at": events[-1]["timestamp"] if events else None,
        "states": dict(Counter(event["state"] for event in events)),
    }
    if events:
        by_step: dict[str, Counter[str]] = defaultdict(Counter)
        profiles: set[str] = set()
        modes: set[str] = set()
        models: set[str] = set()
        lrs: set[str] = set()
        for event in events:
            by_step[event["label"]][event["state"]] += 1
            detail = event["details"]
            for key, target in (("profile", profiles), ("mode", modes), ("models", models), ("lr", lrs)):
                value = detail.get(key)
                if value:
                    target.add(value)
        summary["steps"] = {step: dict(counts) for step, counts in sorted(by_step.items())}
        summary["profiles_seen"] = sorted(profiles)
        summary["modes_seen"] = sorted(modes)
        summary["models_seen"] = sorted(models)
        summary["learning_rates_seen"] = sorted(lrs)
    return events, summary


def parse_detail_fields(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for chunk in text.split(";"):
        if "=" in chunk:
            key, value = chunk.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def scan_fasta(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "file": path.name,
        "sequence_count": 0,
        "first_headers": [],
        "organisms_seen": [],
        "taxonomy_ids_seen": [],
        "is_entrapment": "entrap" in path.name.lower(),
        "is_swiss_reviewed": bool(re.search(r"(revi|swiss)", path.name, re.I)),
    }
    organisms: Counter[str] = Counter()
    taxids: Counter[str] = Counter()
    try:
        with path.open("r", errors="replace") as handle:
            for line in handle:
                if not line.startswith(">"):
                    continue
                info["sequence_count"] += 1
                header = line.strip()
                if len(info["first_headers"]) < 3:
                    info["first_headers"].append(header[:220])
                os_match = re.search(r"\bOS=([^=]+?)(?:\sOX=|\sGN=|\sPE=|\sSV=|$)", header)
                ox_match = re.search(r"\bOX=(\d+)", header)
                if os_match:
                    organisms[os_match.group(1).strip()] += 1
                if ox_match:
                    taxids[ox_match.group(1)] += 1
    except OSError:
        pass
    info["organisms_seen"] = [name for name, _ in organisms.most_common(5)]
    info["taxonomy_ids_seen"] = [taxid for taxid, _ in taxids.most_common(5)]
    return info


def infer_organism(run_name: str, dataset_meta: dict[str, Any], fasta_infos: list[dict[str, Any]]) -> dict[str, Any]:
    if dataset_meta.get("organism"):
        return {
            "name": dataset_meta.get("organism"),
            "taxonomy_id": str(dataset_meta.get("taxonomy_id") or ""),
            "source": "dataset yaml",
            "slug": slugify(dataset_meta.get("organism", "unknown")),
        }
    for fasta in fasta_infos:
        if fasta.get("organisms_seen"):
            name = fasta["organisms_seen"][0]
            taxid = fasta.get("taxonomy_ids_seen", [""])[0] if fasta.get("taxonomy_ids_seen") else ""
            return {"name": name, "taxonomy_id": taxid, "source": "FASTA header", "slug": slugify(name)}
    lowered = run_name.lower()
    for key, (name, taxid) in ORGANISM_ALIASES.items():
        if key in lowered:
            return {"name": name, "taxonomy_id": taxid, "source": "run name heuristic", "slug": slugify(name)}
    return {"name": "unknown", "taxonomy_id": "", "source": "not found", "slug": "unknown"}


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "unknown"


def parse_start_commands(run_dir: Path) -> dict[str, Any]:
    start_files = sorted(
        path
        for path in run_dir.rglob("startcmd*.txt")
        if "run_info" not in path.parts and ".bak" not in path.name
    )
    diann_commands: list[dict[str, Any]] = []
    wrappers: list[dict[str, Any]] = []
    raw_files: set[str] = set()
    fixed_mods: set[str] = set()
    var_mods: set[str] = set()
    flags: Counter[str] = Counter()
    for start_file in start_files:
        text = read_text(start_file)
        for line in command_lines(text):
            tokens = shell_tokens(line)
            if not tokens:
                continue
            rel = str(start_file.relative_to(run_dir))
            if tokens[0].startswith("diann-") or "DIA-NN.exe" in tokens[0]:
                fixed_mods.update(option_values(tokens, "--fixed-mod"))
                var_mods.update(option_values(tokens, "--var-mod"))
                for token in tokens:
                    if token.startswith("--"):
                        flags[token] += 1
                diann_commands.append(
                    {
                        "source": rel,
                        "diann_bin": tokens[0],
                        "threads": option_value(tokens, "--threads"),
                        "fasta": option_value(tokens, "--fasta"),
                        "out": option_value(tokens, "--out"),
                        "out_lib": option_value(tokens, "--out-lib"),
                        "qvalue": option_value(tokens, "--qvalue"),
                        "fixed_mods": option_values(tokens, "--fixed-mod"),
                        "var_mods": option_values(tokens, "--var-mod"),
                        "raw_count": len(option_values(tokens, "--f")),
                    }
                )
                raw_files.update(option_values(tokens, "--f"))
            else:
                wrappers.append(
                    {
                        "source": rel,
                        "script": Path(tokens[1] if tokens[0] == "bash" and len(tokens) > 1 else tokens[0]).name,
                        "profile": option_value(tokens, "--profile"),
                        "out_sub": option_value(tokens, "--out-sub"),
                        "raw_count": len(option_values(tokens, "--f")),
                    }
                )
                raw_files.update(option_values(tokens, "--f"))
    return {
        "start_file_count": len(start_files),
        "diann_command_count": len(diann_commands),
        "wrapper_command_count": len(wrappers),
        "diann_commands": diann_commands,
        "wrapper_commands": wrappers,
        "raw_files": sorted(raw_files),
        "raw_file_count": len(raw_files),
        "raw_file_stems_sample": sorted({Path(file).stem for file in raw_files})[:12],
        "fixed_mods": sorted(fixed_mods),
        "var_mods": sorted(var_mods),
        "common_diann_flags": [flag for flag, _ in flags.most_common(30)],
    }


def parse_planned_searches(all_startcmds: str) -> dict[str, Any]:
    searches = []
    for line in command_lines(all_startcmds):
        if "step2_tunelib.sh" not in line and "step5_quant.sh" not in line and "make_cal_lib.sh" not in line:
            continue
        tokens = shell_tokens(line)
        script = Path(tokens[1] if tokens and tokens[0] == "bash" and len(tokens) > 1 else tokens[0]).name
        raw_files = option_values(tokens, "--f")
        searches.append(
            {
                "script": script,
                "profile": option_value(tokens, "--profile"),
                "out_sub": option_value(tokens, "--out-sub"),
                "lib": option_value(tokens, "--lib"),
                "ref": option_value(tokens, "--ref"),
                "raw_count": len(raw_files),
                "raw_stems": [Path(file).stem for file in raw_files],
            }
        )
    by_kind: dict[str, Any] = defaultdict(lambda: {"runs": 0, "raw_files": 0, "profiles": set(), "out_subs": set()})
    for search in searches:
        bucket = by_kind[search["script"]]
        bucket["runs"] += 1
        bucket["raw_files"] += search["raw_count"]
        if search.get("profile"):
            bucket["profiles"].add(search["profile"])
        if search.get("out_sub"):
            bucket["out_subs"].add(search["out_sub"])
    return {
        "planned_searches": searches,
        "planned_search_summary": {
            key: {
                "runs": value["runs"],
                "raw_files": value["raw_files"],
                "profiles": sorted(value["profiles"]),
                "out_subs": sorted(value["out_subs"]),
            }
            for key, value in sorted(by_kind.items())
        },
    }


def classify_status(run_dir: Path, progress_summary: dict[str, Any]) -> dict[str, Any]:
    states = progress_summary.get("states", {})
    fail_dirs = [str(path.relative_to(run_dir)) for path in run_dir.glob("fail*") if path.is_dir()]
    has_quant = any((run_dir / "05_quant").glob("**/final_report.stats.tsv"))
    has_step_output = any(
        (run_dir / step).exists()
        for step in ("01_predlib", "02_tunelib", "03_tune_models", "04_refined_lib", "04_refined_lib_no_presearch", "cal_lib")
    )
    if states.get("FAIL") or fail_dirs:
        status = "failed"
    elif has_quant:
        status = "complete"
    elif progress_summary.get("event_count", 0) == 0:
        status = "partial" if has_step_output else "unknown"
    elif "5 quant searches" in progress_summary.get("steps", {}) or "MA/QC report" in progress_summary.get("steps", {}):
        status = "complete"
    else:
        status = "partial"
    return {"status": status, "fail_dirs": fail_dirs}


def collect_run(run_dir: Path, version: str, dataset_metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    all_startcmds_path = run_dir / "all_startcmds.txt"
    all_startcmds = read_text(all_startcmds_path)
    invocation = parse_run_invocation(all_startcmds)
    progress_events, progress_summary = parse_progress(run_dir / "progress.txt")
    start_info = parse_start_commands(run_dir)
    fasta_paths = {path for path in sorted(run_dir.glob("*.fasta"))}
    for command in start_info.get("diann_commands", []):
        fasta = command.get("fasta")
        resolved = resolve_fasta_path(fasta)
        if resolved:
            fasta_paths.add(resolved)
    if invocation.get("fasta_abs") and Path(invocation["fasta_abs"]).exists():
        fasta_paths.add(Path(invocation["fasta_abs"]))
    fasta_infos = [scan_fasta(path) for path in sorted(fasta_paths)]
    dataset_id = invocation.get("dataset")
    dataset_meta = dataset_metadata.get(dataset_id or "", {})
    organism = infer_organism(run_dir.name, dataset_meta, fasta_infos)
    planned = parse_planned_searches(all_startcmds)
    status = classify_status(run_dir, progress_summary)
    latest_mtime = max((path.stat().st_mtime for path in run_dir.rglob("*") if path.is_file()), default=run_dir.stat().st_mtime)
    created_hint = progress_summary.get("started_at")
    set_values = invocation.get("set_values", {})
    mod_config_value = set_values.get("MOD_CONFIG") or invocation.get("mods_config")
    return {
        "run_id": f"{version}/{run_dir.name}",
        "experiment": invocation.get("expname") or run_dir.name,
        "run_dir": str(run_dir),
        "pipeline_version": version,
        "dataset": dataset_id,
        "dataset_meta": dataset_meta,
        "organism": organism,
        "status": status["status"],
        "fail_dirs": status["fail_dirs"],
        "started_at": created_hint,
        "ended_at": progress_summary.get("ended_at"),
        "last_file_change": dt.datetime.fromtimestamp(latest_mtime).isoformat(timespec="seconds"),
        "run_all": {
            key: value
            for key, value in invocation.items()
            if key not in {"set_values", "expname", "dataset"}
        },
        "settings": {
            "threads": set_values.get("THREADS"),
            "tune_lr": set_values.get("TUNE_LR"),
            "step4_lr_select": set_values.get("STEP4_LR_SELECT"),
            "step4_model_targets": set_values.get("STEP4_MODEL_TARGETS"),
            "dataset_type": set_values.get("DATASET_TYPE"),
            "mod_config": mod_config_value,
            "no_cal_lib": invocation.get("no_cal_lib"),
            "fasta_kind": invocation.get("fasta_kind"),
            "ram_log": invocation.get("ram_log"),
        },
        "mod_config_details": parse_mod_config(mod_config_value),
        "fasta": fasta_infos,
        "progress": progress_summary,
        "events": progress_events,
        "start_commands": start_info,
        **planned,
    }


def resolve_fasta_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return path
    for directory in FASTA_FALLBACK_DIRS:
        candidate = directory / path.name
        if candidate.exists():
            return candidate
    return None


def discover_runs(root: Path, versions: list[str]) -> list[tuple[Path, str]]:
    runs = []
    for version in versions:
        version_dir = root / version
        if not version_dir.exists():
            continue
        for run_dir in sorted(path for path in version_dir.iterdir() if path.is_dir()):
            if (run_dir / "progress.txt").exists() or (run_dir / "all_startcmds.txt").exists() or (run_dir / "01_predlib").exists():
                runs.append((run_dir, version))
    return runs


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_sqlite(path: Path, runs: list[dict[str, Any]]) -> None:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                experiment TEXT,
                pipeline_version TEXT,
                organism TEXT,
                taxonomy_id TEXT,
                dataset TEXT,
                status TEXT,
                started_at TEXT,
                ended_at TEXT,
                run_dir TEXT,
                fasta_files TEXT,
                mod_config TEXT,
                profiles TEXT,
                raw_file_count INTEGER,
                details_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE events (
                run_id TEXT,
                timestamp TEXT,
                step TEXT,
                label TEXT,
                state TEXT,
                details_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE searches (
                run_id TEXT,
                script TEXT,
                profile TEXT,
                out_sub TEXT,
                raw_count INTEGER,
                raw_stems TEXT
            )
            """
        )
        for run in runs:
            conn.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run["run_id"],
                    run["experiment"],
                    run["pipeline_version"],
                    run["organism"]["name"],
                    run["organism"].get("taxonomy_id"),
                    run.get("dataset"),
                    run["status"],
                    run.get("started_at"),
                    run.get("ended_at"),
                    run["run_dir"],
                    ", ".join(fasta["file"] for fasta in run["fasta"]),
                    run["settings"].get("mod_config"),
                    ", ".join(run["progress"].get("profiles_seen", [])),
                    run["start_commands"].get("raw_file_count"),
                    json.dumps(run, sort_keys=True),
                ),
            )
            for event in run["events"]:
                conn.execute(
                    "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        run["run_id"],
                        event["timestamp"],
                        event["step"],
                        event["label"],
                        event["state"],
                        json.dumps(event["details"], sort_keys=True),
                    ),
                )
            for search in run["planned_searches"]:
                conn.execute(
                    "INSERT INTO searches VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        run["run_id"],
                        search["script"],
                        search.get("profile"),
                        search.get("out_sub"),
                        search["raw_count"],
                        ", ".join(search["raw_stems"]),
                    ),
                )
        conn.commit()
    finally:
        conn.close()


def markdown_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|")


def status_badge(status: str) -> str:
    return STATUS_BADGES.get(status, f"⚪ {status}")


def format_mod_entry(entry: dict[str, Any]) -> str:
    labels = ", ".join(entry.get("site_labels", [])) or entry.get("sites", "")
    comment = entry.get("comment")
    suffix = f"; {comment}" if comment else ""
    identity = entry["unimod"] if entry["name"] == entry["unimod"] else f"{entry['name']} ({entry['unimod']})"
    return f"{identity} (+{entry['mass']}, {labels}{suffix})"


def format_compact_mod_entry(entry: dict[str, Any]) -> str:
    labels = "+".join(entry.get("site_labels", [])) or entry.get("sites", "")
    identity = entry["unimod"] if entry["name"] == entry["unimod"] else f"{entry['name']} {entry['unimod']}"
    return f"{identity} +{entry['mass']} on {labels}"


def summarize_mods(run: dict[str, Any], *, compact: bool = True) -> str:
    details = run.get("mod_config_details", {})
    formatter = format_compact_mod_entry if compact else format_mod_entry
    fixed = [formatter(entry) for entry in details.get("fixed", [])]
    variable = [formatter(entry) for entry in details.get("variable", [])]
    if not fixed and not variable:
        fixed = run["start_commands"].get("fixed_mods", [])
        variable = run["start_commands"].get("var_mods", [])
    chunks = []
    if fixed:
        chunks.append("fixed: " + ("; ".join(fixed) if not compact else "; ".join(fixed)))
    if variable:
        chunks.append("variable: " + ("; ".join(variable) if not compact else "; ".join(variable)))
    if not chunks:
        return details.get("config") or run["settings"].get("mod_config") or "unknown"
    config = details.get("config") or run["settings"].get("mod_config")
    prefix = f"{config}: " if config else ""
    return prefix + " | ".join(chunks)


def write_overview(path: Path, runs: list[dict[str, Any]], generated_at: str) -> None:
    status_counts = Counter(run["status"] for run in runs)
    organism_counts = Counter(run["organism"]["name"] for run in runs)
    lines = [
        "# DIA-NN Run Info",
        "",
        f"Generated: {generated_at}",
        "",
        "This folder is generated by `diaNN_pipline/bin/update_run_info.py` from run logs, progress files and start commands.",
        "The original run folders are read-only inputs for this catalogue.",
        "",
        "## Files",
        "",
        "- `diann_runs.sqlite`: queryable lightweight database",
        "- `runs.json`: complete extracted metadata",
        "- `runs.csv`: compact table for spreadsheets",
        "- `organisms/*.md`: human-readable run notes grouped by organism",
        "",
        "## Summary",
        "",
        f"- Runs: {len(runs)}",
        f"- Status: {', '.join(f'{key}={value}' for key, value in sorted(status_counts.items()))}",
        f"- Organisms: {', '.join(f'{key}={value}' for key, value in sorted(organism_counts.items()))}",
        "",
        "## Compact Run Table",
        "",
        "| run | status | organism | dataset | FASTA | mods | profiles | started | ended |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in sorted(runs, key=lambda item: (item["organism"]["name"], item["pipeline_version"], item["experiment"])):
        fasta = ", ".join(item["file"] for item in run["fasta"]) or ""
        profiles = ", ".join(run["progress"].get("profiles_seen", []))
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(value)
                for value in [
                    run["run_id"],
                    status_badge(run["status"]),
                    run["organism"]["name"],
                    run.get("dataset"),
                    fasta,
                    summarize_mods(run),
                    profiles,
                    run.get("started_at"),
                    run.get("ended_at"),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    fields = [
        "run_id",
        "experiment",
        "pipeline_version",
        "status",
        "organism",
        "taxonomy_id",
        "dataset",
        "dataset_type",
        "fasta_files",
        "fasta_sequence_counts",
        "mod_config",
        "mod_details",
        "fixed_mods",
        "var_mods",
        "profiles_seen",
        "models_seen",
        "learning_rates_seen",
        "raw_file_count",
        "started_at",
        "ended_at",
        "run_dir",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "run_id": run["run_id"],
                    "experiment": run["experiment"],
                    "pipeline_version": run["pipeline_version"],
                    "status": run["status"],
                    "organism": run["organism"]["name"],
                    "taxonomy_id": run["organism"].get("taxonomy_id"),
                    "dataset": run.get("dataset"),
                    "dataset_type": run["settings"].get("dataset_type"),
                    "fasta_files": "; ".join(fasta["file"] for fasta in run["fasta"]),
                    "fasta_sequence_counts": "; ".join(str(fasta["sequence_count"]) for fasta in run["fasta"]),
                    "mod_config": run["settings"].get("mod_config"),
                    "mod_details": summarize_mods(run, compact=False),
                    "fixed_mods": "; ".join(run["start_commands"].get("fixed_mods", [])),
                    "var_mods": "; ".join(run["start_commands"].get("var_mods", [])),
                    "profiles_seen": "; ".join(run["progress"].get("profiles_seen", [])),
                    "models_seen": "; ".join(run["progress"].get("models_seen", [])),
                    "learning_rates_seen": "; ".join(run["progress"].get("learning_rates_seen", [])),
                    "raw_file_count": run["start_commands"].get("raw_file_count"),
                    "started_at": run.get("started_at"),
                    "ended_at": run.get("ended_at"),
                    "run_dir": run["run_dir"],
                }
            )


def write_organism_markdown(directory: Path, runs: list[dict[str, Any]], generated_at: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.md"):
        old.unlink()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["organism"]["slug"]].append(run)
    for slug, org_runs in sorted(grouped.items()):
        org_runs.sort(key=lambda item: (item["pipeline_version"], item["experiment"]))
        organism = org_runs[0]["organism"]
        lines = [
            f"# {organism['name']}",
            "",
            f"Generated: {generated_at}",
            "",
            f"Taxonomy ID: {organism.get('taxonomy_id') or 'unknown'}",
            "",
            "## Runs",
            "",
            "| run | status | dataset | FASTA | mods | searches | profiles/models | time span |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for run in org_runs:
            fasta = ", ".join(
                f"{item['file']} ({item['sequence_count']} seqs)" if item.get("sequence_count") else item["file"]
                for item in run["fasta"]
            )
            searches = summarize_searches(run)
            profiles_models = summarize_profiles_models(run)
            span = f"{run.get('started_at') or '?'} to {run.get('ended_at') or '?'}"
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in [
                        run["run_id"],
                        status_badge(run["status"]),
                        run.get("dataset"),
                        fasta,
                        summarize_mods(run),
                        searches,
                        profiles_models,
                        span,
                    ]
                )
                + " |"
            )
        lines.extend(["", "## Details", ""])
        for run in org_runs:
            lines.extend(run_detail_markdown(run))
        (directory / f"{slug}.md").write_text("\n".join(lines) + "\n")


def summarize_searches(run: dict[str, Any]) -> str:
    parts = []
    for script, summary in run["planned_search_summary"].items():
        parts.append(f"{script}: {summary['runs']} cmds/{summary['raw_files']} files")
    return "; ".join(parts)


def summarize_profiles_models(run: dict[str, Any]) -> str:
    profiles = ", ".join(run["progress"].get("profiles_seen", [])) or run["run_all"].get("profile_request") or ""
    models = ", ".join(run["progress"].get("models_seen", []))
    lrs = ", ".join(run["progress"].get("learning_rates_seen", []))
    chunks = [chunk for chunk in [f"profiles={profiles}" if profiles else "", f"models={models}" if models else "", f"lr={lrs}" if lrs else ""] if chunk]
    return "; ".join(chunks)


def format_mod_list(run: dict[str, Any], kind: str) -> str:
    entries = run.get("mod_config_details", {}).get(kind, [])
    if entries:
        return "; ".join(format_mod_entry(entry) for entry in entries)
    command_key = "fixed_mods" if kind == "fixed" else "var_mods"
    return ", ".join(run["start_commands"].get(command_key, [])) or "not detected"


def run_detail_markdown(run: dict[str, Any]) -> list[str]:
    lines = [
        f"### {run['run_id']}",
        "",
        f"- Status: {status_badge(run['status'])}",
        f"- Directory: `{run['run_dir']}`",
        f"- Dataset: {run.get('dataset') or 'unknown'}; type: {run['settings'].get('dataset_type') or 'unknown'}",
        f"- FASTA: {', '.join(item['file'] for item in run['fasta']) or 'unknown'}",
        f"- FASTA note: {fasta_note(run)}",
        f"- Mod config: {run['settings'].get('mod_config') or 'unknown'}",
        f"- Fixed mods: {format_mod_list(run, 'fixed')}",
        f"- Variable mods: {format_mod_list(run, 'variable')}",
        f"- Raw files detected: {run['start_commands'].get('raw_file_count', 0)} ({', '.join(run['start_commands'].get('raw_file_stems_sample', []))})",
        f"- Profiles/models/LR: {summarize_profiles_models(run) or 'not detected'}",
    ]
    if run.get("fail_dirs"):
        lines.append(f"- Fail folders: {', '.join(run['fail_dirs'])}")
    if run["progress"].get("steps"):
        step_bits = []
        for step, counts in run["progress"]["steps"].items():
            step_bits.append(f"{step}: " + ", ".join(f"{state}={count}" for state, count in sorted(counts.items())))
        lines.append(f"- Progress: {'; '.join(step_bits)}")
    if run["planned_search_summary"]:
        lines.append(f"- Planned searches: {summarize_searches(run)}")
    lines.append("")
    return lines


def fasta_note(run: dict[str, Any]) -> str:
    notes = []
    for fasta in run["fasta"]:
        bits = []
        if fasta.get("is_swiss_reviewed"):
            bits.append("Swiss/reviewed-looking")
        if fasta.get("is_entrapment"):
            bits.append("entrapment combined")
        if fasta.get("organisms_seen"):
            bits.append("headers: " + ", ".join(fasta["organisms_seen"][:2]))
        notes.append(f"{fasta['file']} ({'; '.join(bits) if bits else 'standard-looking'})")
    return "; ".join(notes) if notes else "no FASTA file found in run root"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="DIA-NN output root containing version folders")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Generated catalogue output directory")
    parser.add_argument("--versions", nargs="+", default=["2_3_2", "2_5_0"], help="Version folders to scan")
    args = parser.parse_args()

    dataset_metadata = load_dataset_metadata()
    runs = [collect_run(run_dir, version, dataset_metadata) for run_dir, version in discover_runs(args.root, args.versions)]
    runs.sort(key=lambda item: (item["organism"]["name"], item["pipeline_version"], item["experiment"]))

    args.out.mkdir(parents=True, exist_ok=True)
    generated_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    payload = {"generated_at": generated_at, "root": str(args.root), "runs": runs}
    write_json(args.out / "runs.json", payload)
    write_csv(args.out / "runs.csv", runs)
    write_sqlite(args.out / "diann_runs.sqlite", runs)
    write_overview(args.out / "README.md", runs, generated_at)
    write_organism_markdown(args.out / "organisms", runs, generated_at)
    print(f"Wrote {len(runs)} runs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
