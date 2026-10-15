#!/usr/bin/env python3
"""Versioned catalog and reproducible builder for Carafe2/DIA-NN fine-tuning data.

The input unit is a completed Carafe ``carafe_output`` directory.  A build
combines one or more compatible units into a standard Carafe2 input bundle and
an identical DIA-NN RT input bundle.  It never modifies source exports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from shutil import copy2
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:  # Supports both ``python fine_tune_catalog.py`` and package imports in tests.
    from .rt_alignment import apply_rt_target, fit_target_alignment
    from .annotate_rt_trypticity import (
        ANNOTATION_COLUMNS,
        EXPECTED_CUT,
        EXPECTED_SEMI,
        annotate_sequences,
        load_fasta,
        parse_contract,
        report_protein_mapping,
    )
except ImportError:  # pragma: no cover - exercised by the command-line entry point.
    from rt_alignment import apply_rt_target, fit_target_alignment
    from annotate_rt_trypticity import (
        ANNOTATION_COLUMNS,
        EXPECTED_CUT,
        EXPECTED_SEMI,
        annotate_sequences,
        load_fasta,
        parse_contract,
        report_protein_mapping,
    )


RT_FILE = "rt_train_data.tsv"
PSM_FILE = "psm_pdv.txt"
FRAGMENT_FILE = "fragment_intensity_df.tsv"
FRAGMENT_VALID_FILE = "fragment_intensity_valid.tsv"
META_FILE = "meta.json"
RT_COLUMNS = {"sequence", "mods", "mod_sites", "rt", "rt_norm"}
PSM_INDEX_COLUMNS = ("frag_start_idx", "frag_stop_idx")
ALIGNED_RT_METHODS = {"anchor_irt", "linear", "chronologer_hi"}
TRYPTICITY_VERSION = "fasta_backed_dia_nn_v1"


@dataclass(frozen=True)
class Source:
    dataset_id: str
    output_dir: Path
    metadata: dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog(path: Path) -> dict[str, Any]:
    catalog = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(catalog, dict) or "datasets" not in catalog or "builds" not in catalog:
        raise ValueError("Catalog requires top-level 'datasets' and 'builds' lists")
    ids = [str(item.get("id", "")) for item in catalog["datasets"]]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Every dataset requires a unique non-empty id")
    return catalog


def source_rt_max(source: Source) -> float:
    override = source.metadata.get("rt_max_minutes")
    if override is not None:
        return float(override)
    meta_path = source.output_dir / META_FILE
    if not meta_path.is_file():
        raise ValueError(f"{source.dataset_id}: missing {META_FILE}; set rt_max_minutes explicitly in the catalog")
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    maxima = [float(value["rt_max"]) for value in meta.values() if isinstance(value, dict) and "rt_max" in value]
    if not maxima or max(maxima) <= 0:
        raise ValueError(f"{source.dataset_id}: no valid rt_max in {meta_path}")
    return max(maxima)


def selected_sources(catalog: dict[str, Any], build_id: str) -> tuple[dict[str, Any], list[Source]]:
    build = next((item for item in catalog["builds"] if item.get("id") == build_id), None)
    if build is None:
        raise ValueError(f"Unknown build id: {build_id}")
    by_id = {item["id"]: item for item in catalog["datasets"]}
    wanted = build.get("datasets", [])
    if not wanted:
        raise ValueError(f"Build {build_id} has no datasets")
    missing = [item for item in wanted if item not in by_id]
    if missing:
        raise ValueError(f"Build {build_id} references unknown datasets: {missing}")
    sources = []
    for dataset_id in wanted:
        entry = by_id[dataset_id]
        if entry.get("enabled", True) is False:
            raise ValueError(f"Build {build_id} references disabled dataset {dataset_id}")
        output_dir = Path(entry["carafe_output_dir"])
        sources.append(Source(dataset_id, output_dir, entry))
    return build, sources


def validate_source(source: Source, *, require_ms2: bool) -> dict[str, Any]:
    required = [RT_FILE] + ([PSM_FILE, FRAGMENT_FILE] if require_ms2 else [])
    missing = [str(source.output_dir / name) for name in required if not (source.output_dir / name).is_file()]
    if missing:
        raise ValueError(f"{source.dataset_id}: missing source files:\n" + "\n".join(missing))
    rt = pd.read_csv(source.output_dir / RT_FILE, sep="\t", dtype={"mods": str, "mod_sites": str})
    missing_columns = RT_COLUMNS - set(rt.columns)
    if missing_columns:
        raise ValueError(f"{source.dataset_id}: {RT_FILE} lacks {sorted(missing_columns)}")
    target = pd.to_numeric(rt["rt_norm"], errors="coerce")
    if target.isna().any() or not np.isfinite(target).all() or ((target < -1e-6) | (target > 1 + 1e-6)).any():
        raise ValueError(f"{source.dataset_id}: rt_norm must be finite and within [0, 1]")
    report: dict[str, Any] = {
        "id": source.dataset_id,
        "original_experiment_name": source.metadata.get("original_experiment_name", ""),
        "source_step2_experiment": source.metadata.get("source_step2_experiment", ""),
        "carafe_job_dir": source.metadata.get("carafe_job_dir", ""),
        "diann_step2_report": source.metadata.get("diann_step2_report", ""),
        "train_mzml_dir": source.metadata.get("train_mzml_dir", ""),
        "organism": source.metadata.get("organism", ""),
        "comment": source.metadata.get("comment", ""),
        "carafe_output_dir": str(source.output_dir.resolve()),
        "rt_rows": int(len(rt)),
        "rt_unique_peptidoforms": int(rt[["sequence", "mods", "mod_sites"]].drop_duplicates().shape[0]),
        "rt_max_minutes": source_rt_max(source),
        "rt_sha256": sha256(source.output_dir / RT_FILE),
    }
    if require_ms2:
        psm = pd.read_csv(source.output_dir / PSM_FILE, sep="\t", dtype={"mods": str, "mod_sites": str})
        fragment = pd.read_csv(source.output_dir / FRAGMENT_FILE, sep="\t")
        if not set(PSM_INDEX_COLUMNS).issubset(psm.columns):
            raise ValueError(f"{source.dataset_id}: {PSM_FILE} lacks {PSM_INDEX_COLUMNS}")
        starts = pd.to_numeric(psm["frag_start_idx"], errors="raise").astype(int)
        stops = pd.to_numeric(psm["frag_stop_idx"], errors="raise").astype(int)
        if (starts < 0).any() or (stops < starts).any() or stops.max() != len(fragment):
            raise ValueError(f"{source.dataset_id}: invalid fragment index contract in {PSM_FILE}")
        report.update({
            "psm_rows": int(len(psm)), "fragment_rows": int(len(fragment)),
            "psm_sha256": sha256(source.output_dir / PSM_FILE),
            "fragment_sha256": sha256(source.output_dir / FRAGMENT_FILE),
            "has_fragment_valid": (source.output_dir / FRAGMENT_VALID_FILE).is_file(),
        })
    return report


def discover_job(job_dir: Path) -> dict[str, Any]:
    """Read a prepared Carafe job and emit a catalog-entry template.

    The job is the durable bridge from an external DIA-NN search result to the
    Carafe exports: its manifest records the original Step-2 report and its
    input directory contains the staged report and training mzML files.
    """
    job_dir = job_dir.resolve()
    manifest_path = job_dir / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    source_exp = (manifest.get("source_experiment", {}) or {}).get("name", "")
    report = (manifest.get("inputs", {}) or {}).get("diann_step2_report", "")
    if not report:
        candidate = job_dir / "input" / "diann_step2_report.parquet"
        report = str(candidate) if candidate.is_file() else ""
    output = job_dir / "carafe_output"
    return {
        "id": "replace_with_permanent_dataset_id",
        "enabled": True,
        "original_experiment_name": source_exp or "replace_with_original_experiment_name",
        "source_step2_experiment": source_exp,
        "organism": "replace_with_organism",
        "gradient_minutes": "replace_with_gradient_minutes",
        "instrument": "replace_with_instrument",
        "comment": "replace_with_comment",
        "carafe_job_dir": str(job_dir),
        "diann_step2_report": report,
        "train_mzml_dir": str(job_dir / "input" / "train_mzml"),
        "carafe_output_dir": str(output),
    }


def ensure_compatible(
    reports: list[dict[str, Any]], *, normalization: dict[str, Any], require_ms2: bool
) -> None:
    maxima = [item["rt_max_minutes"] for item in reports]
    policy = normalization.get("policy", "strict_same_global_rt_max")
    tolerance = float(normalization.get("rt_max_tolerance_minutes", 0.25))
    if policy == "strict_same_global_rt_max" and max(maxima) - min(maxima) > tolerance:
        values = ", ".join(f"{item['id']}={item['rt_max_minutes']:.4f}" for item in reports)
        raise ValueError("Incompatible RT normalization maxima (strict policy): " + values)
    if policy == "rebase_to_global_rt_max":
        target = float(normalization.get("target_rt_max_minutes", 0))
        if target <= 0:
            raise ValueError("rebase_to_global_rt_max requires a positive target_rt_max_minutes")
        if max(maxima) - target > tolerance:
            raise ValueError(
                "target_rt_max_minutes must be at least every source rt_max; "
                f"target={target:.4f}, largest source={max(maxima):.4f}"
            )
    if policy == "aligned":
        method = str(normalization.get("method", ""))
        if method not in ALIGNED_RT_METHODS:
            raise ValueError(
                "aligned RT policy requires method one of "
                + ", ".join(sorted(ALIGNED_RT_METHODS))
            )
        reference_minutes = float(normalization.get("reference_minutes", 0))
        if reference_minutes <= 0:
            raise ValueError("aligned RT policy requires positive reference_minutes")
        if method in {"anchor_irt", "linear"}:
            reference_source = str(normalization.get("reference_source", ""))
            source_ids = {item["id"] for item in reports}
            if reference_source not in source_ids:
                raise ValueError(
                    "aligned anchor/linear policy requires reference_source from selected datasets"
                )
    if policy not in {"strict_same_global_rt_max", "rebase_to_global_rt_max", "aligned"}:
        raise ValueError(f"Unsupported RT normalization policy: {policy}")
    if require_ms2:
        validity = {item["has_fragment_valid"] for item in reports}
        if len(validity) != 1:
            raise ValueError("All MS2 sources must either all have or all lack fragment_intensity_valid.tsv")


def _read_rt_observations(sources: list[Source]) -> dict[str, pd.DataFrame]:
    """Read source RT tables while retaining each raw observation and its provenance."""
    frames: dict[str, pd.DataFrame] = {}
    for source in sources:
        frame = pd.read_csv(source.output_dir / RT_FILE, sep="\t", dtype={"mods": str, "mod_sites": str})
        frame["mods"] = frame["mods"].fillna("").astype(str)
        frame["mod_sites"] = frame["mod_sites"].fillna("").astype(str)
        frame["rt"] = pd.to_numeric(frame["rt"], errors="raise")
        frame["source_rt_norm"] = pd.to_numeric(frame["rt_norm"], errors="raise")
        frame["source_dataset_id"] = source.dataset_id
        frame["source_organism"] = str(source.metadata.get("organism", "unknown"))
        frames[source.dataset_id] = frame
    return frames


def _annotate_trypticity(
    observations: pd.DataFrame, sources: list[Source], *, enabled: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach strict source-specific digestion annotations from report + FASTA."""
    if not enabled:
        return observations, {"enabled": False}
    specs = {source.dataset_id: source.metadata.get("trypticity", {}) for source in sources}
    missing = [source_id for source_id, spec in specs.items() if not isinstance(spec, dict) or not spec]
    if missing:
        raise ValueError(
            "Trypticity annotation is enabled but these datasets lack a trypticity report/FASTA spec: "
            + ", ".join(missing)
        )
    report_cache: dict[Path, dict[str, tuple[str, ...]]] = {}
    fasta_cache: dict[Path, dict[str, tuple[str, ...]]] = {}
    annotation_frames, source_details = [], []
    for source_id, spec in specs.items():
        report = Path(spec.get("report", ""))
        fasta = Path(spec.get("fasta", ""))
        contract_path = Path(spec.get("digestion_contract", ""))
        missing_paths = [str(path) for path in (report, fasta, contract_path) if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(f"{source_id}: missing trypticity inputs: " + ", ".join(missing_paths))
        contract = parse_contract(contract_path)
        if contract.get("DIANN_CUT") != EXPECTED_CUT or contract.get("DIANN_SEMI") != EXPECTED_SEMI:
            raise ValueError(
                f"{source_id}: expected semi-specific {EXPECTED_CUT} digestion contract, "
                f"found DIANN_CUT={contract.get('DIANN_CUT')!r}, DIANN_SEMI={contract.get('DIANN_SEMI')!r}"
            )
        if report not in report_cache:
            report_cache[report] = report_protein_mapping(report)
        if fasta not in fasta_cache:
            fasta_cache[fasta] = load_fasta(fasta)
        proteins = report_cache[report]
        fasta_records = fasta_cache[fasta]
        source_rows = observations.loc[observations["source_dataset_id"].eq(source_id), "sequence"]
        annotation = annotate_sequences(source_rows.astype(str), proteins, fasta_records, source_id)
        annotation = annotation.rename(columns={"source_organism": "source_dataset_id"})
        annotation_frames.append(annotation)
        source_details.append({
            "source_dataset_id": source_id,
            "report": str(report.resolve()), "report_sha256": sha256(report),
            "fasta": str(fasta.resolve()), "fasta_sha256": sha256(fasta),
            "digestion_contract": str(contract_path.resolve()),
            "digestion_contract_sha256": sha256(contract_path),
            "digestion_rule": contract["DIANN_CUT"],
            "semi_specific_search": contract["DIANN_SEMI"] == "1",
            "annotation_sequences": int(len(annotation)),
            "unmapped_sequences": int(annotation["trypticity_detail"].eq("unmapped").sum()),
            "ambiguous_sequences": int(annotation["trypticity_detail"].eq("ambiguous").sum()),
        })
    annotations = pd.concat(annotation_frames, ignore_index=True)
    annotated = observations.merge(
        annotations, on=["source_dataset_id", "sequence"], how="left", validate="many_to_one"
    )
    if annotated["digestion_class"].isna().any():
        raise RuntimeError("Trypticity annotation left unannotated source observations")
    return annotated, {
        "enabled": True,
        "version": TRYPTICITY_VERSION,
        "classification_policy": (
            "Every DIA-NN-reported protein and every matching exact FASTA occurrence; "
            "conflicting protein contexts are ambiguous."
        ),
        "annotation_columns": list(ANNOTATION_COLUMNS),
        "sources": source_details,
    }


def _consensus_text(values: pd.Series, *, conflict: str = "ambiguous") -> str:
    unique = sorted({str(value) for value in values.dropna()})
    return unique[0] if len(unique) == 1 else conflict


def _aggregate_trypticity(observations: pd.DataFrame) -> pd.DataFrame:
    """Summarize source-specific digestion calls without hiding disagreements."""
    keys = ["sequence", "mods", "mod_sites"]
    grouped = observations.groupby(keys, as_index=False, dropna=False)
    result = grouped.agg(
        digestion_class=("digestion_class", lambda values: _consensus_text(values)),
        trypticity_detail=("trypticity_detail", lambda values: _consensus_text(values)),
        digestion_rule=("digestion_rule", lambda values: _consensus_text(values, conflict="mixed")),
        digestion_mapping_count=("digestion_mapping_count", "max"),
        digestion_mapped_protein_count=("digestion_mapped_protein_count", "max"),
        digestion_reported_protein_count=("digestion_reported_protein_count", "max"),
        digestion_mapping_classes=(
            "digestion_mapping_classes",
            lambda values: ";".join(sorted({part for value in values.dropna() for part in str(value).split(";") if part})),
        ),
        trypticity_source_annotation_count=("source_dataset_id", "nunique"),
    )
    result["is_semi_tryptic"] = result["trypticity_detail"].map(
        {"fully_tryptic": False, "semi_nterm": True, "semi_cterm": True}
    ).astype("boolean")
    result["enzymatic_n_terminus"] = result["trypticity_detail"].map(
        {"fully_tryptic": True, "semi_nterm": False, "semi_cterm": True, "non_specific": False}
    ).astype("boolean")
    result["enzymatic_c_terminus"] = result["trypticity_detail"].map(
        {"fully_tryptic": True, "semi_nterm": True, "semi_cterm": False, "non_specific": False}
    ).astype("boolean")
    return result


def _aggregate_target_observations(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Produce one robust target per peptidoform while preserving all source observations.

    A median has a 50% breakdown point: one source-specific alignment residual
    cannot pull the final label arbitrarily far, unlike an arithmetic mean.
    """
    keys = ["sequence", "mods", "mod_sites"]
    grouped = observations.groupby(keys, as_index=False, dropna=False)
    training = grouped.agg(rt=("rt", "median"), rt_norm=("rt_norm", "median"))
    training["rt_train_row_index"] = range(len(training))
    lookup = training[keys + ["rt_train_row_index"]]
    provenance = observations.merge(lookup, on=keys, how="left", validate="many_to_one")
    counts = grouped.size().rename(columns={"size": "observation_count"})
    source_counts = (
        observations.groupby(keys, dropna=False)["source_dataset_id"]
        .nunique()
        .rename("source_dataset_count")
        .reset_index()
    )
    summary = counts.merge(source_counts, on=keys, how="left", validate="one_to_one")
    summary = summary.merge(lookup, on=keys, how="left", validate="one_to_one")
    details = {
        "aggregation": "median_rt_norm_per_sequence_mods_mod_sites",
        "grouping_columns": keys,
        "raw_observation_rows": int(len(observations)),
        "model_training_rows": int(len(training)),
        "collapsed_observation_rows": int(len(observations) - len(training)),
        "multi_observation_peptidoforms": int((summary["observation_count"] > 1).sum()),
        "median_rationale": (
            "Robust consensus target after RT target formation; a single source observation "
            "cannot arbitrarily shift the label as it could with an arithmetic mean."
        ),
    }
    return training.drop(columns=["rt_train_row_index"]), provenance, {**details, "summary": summary}


def copy_rt(
    sources: list[Source], output: Path, *, normalization: dict[str, Any], reports: list[dict[str, Any]],
    trypticity: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Write a model table and auditable raw-observation provenance."""
    policy = normalization.get("policy", "strict_same_global_rt_max")
    target_rt_max = float(normalization.get("target_rt_max_minutes", 0))
    rt_max_by_id = {item["id"]: float(item["rt_max_minutes"]) for item in reports}
    frames_by_id = _read_rt_observations(sources)
    target_metadata: dict[str, Any] = {}
    if policy == "rebase_to_global_rt_max":
        for source_id, frame in frames_by_id.items():
            frame["rt_norm"] = frame["rt"] / target_rt_max
        target_metadata = {
            "target_coordinate": "global_observed_rt_minutes",
            "target_rt_max_minutes": target_rt_max,
            "fit_scope": "none_no_cross_source_alignment",
        }
    elif policy == "aligned":
        method = str(normalization["method"])
        reference_source = str(normalization.get("reference_source", "human"))
        alignment = fit_target_alignment(
            frames_by_id, mode=method, reference_source=reference_source
        )
        transformed, target_metadata = apply_rt_target(
            frames_by_id, mode=method,
            reference_minutes=float(normalization["reference_minutes"]), alignment=alignment,
        )
        frames_by_id = transformed

    observations = pd.concat(frames_by_id.values(), ignore_index=True)
    observations["source_rt_max_minutes"] = observations["source_dataset_id"].map(rt_max_by_id)
    observations["rt_normalization_policy"] = policy
    observations["rt_target_method"] = (
        str(normalization.get("method", "")) if policy == "aligned" else policy
    )
    observations, trypticity_details = _annotate_trypticity(
        observations, sources, enabled=bool((trypticity or {}).get("enabled", False))
    )

    if policy in {"aligned", "rebase_to_global_rt_max"}:
        training, provenance_frame, aggregation_details = _aggregate_target_observations(observations)
        if trypticity_details["enabled"]:
            trypticity_summary = _aggregate_trypticity(observations)
            training = training.merge(
                trypticity_summary, on=["sequence", "mods", "mod_sites"], how="left", validate="one_to_one"
            )
        aggregation_summary = aggregation_details.pop("summary")
        observations.to_csv(output / "rt_training_observations.tsv", sep="\t", index=False)
        aggregation_summary.to_csv(output / "rt_training_aggregation.tsv", sep="\t", index=False)
        preparation = {
            **aggregation_details,
            "target_metadata": target_metadata,
            "observation_table": "rt_training_observations.tsv",
            "aggregation_table": "rt_training_aggregation.tsv",
            "trypticity": trypticity_details,
        }
    else:
        training = observations.drop(columns=["source_rt_norm", "source_dataset_id", "source_organism", "source_rt_max_minutes", "rt_normalization_policy", "rt_target_method"])
        provenance_frame = observations.copy()
        preparation = {
            "aggregation": "none_source_rows_retained", "raw_observation_rows": int(len(observations)),
            "model_training_rows": int(len(training)), "trypticity": trypticity_details,
        }

    training.to_csv(output / RT_FILE, sep="\t", index=False)
    provenance_frame.to_csv(output / "rt_training_provenance.tsv", sep="\t", index=False)
    return training, provenance_frame, preparation


def copy_ms2(sources: list[Source], output: Path) -> dict[str, int]:
    psms, fragments, valids = [], [], []
    offset = 0
    expected_fragment_columns: list[str] | None = None
    has_valid = (sources[0].output_dir / FRAGMENT_VALID_FILE).is_file()
    for source in sources:
        psm = pd.read_csv(source.output_dir / PSM_FILE, sep="\t", dtype={"mods": str, "mod_sites": str})
        fragment = pd.read_csv(source.output_dir / FRAGMENT_FILE, sep="\t")
        if expected_fragment_columns is None:
            expected_fragment_columns = list(fragment.columns)
        elif list(fragment.columns) != expected_fragment_columns:
            raise ValueError(f"{source.dataset_id}: fragment columns differ from first source")
        psm["frag_start_idx"] = pd.to_numeric(psm["frag_start_idx"], errors="raise").astype(int) + offset
        psm["frag_stop_idx"] = pd.to_numeric(psm["frag_stop_idx"], errors="raise").astype(int) + offset
        psm["psm_id"] = range(sum(len(item) for item in psms), sum(len(item) for item in psms) + len(psm))
        psms.append(psm)
        fragments.append(fragment)
        if has_valid:
            valid = pd.read_csv(source.output_dir / FRAGMENT_VALID_FILE, sep="\t")
            if list(valid.columns) != expected_fragment_columns or len(valid) != len(fragment):
                raise ValueError(f"{source.dataset_id}: invalid {FRAGMENT_VALID_FILE}")
            valids.append(valid)
        offset += len(fragment)
    pd.concat(psms, ignore_index=True).to_csv(output / PSM_FILE, sep="\t", index=False)
    pd.concat(fragments, ignore_index=True).to_csv(output / FRAGMENT_FILE, sep="\t", index=False)
    if has_valid:
        pd.concat(valids, ignore_index=True).to_csv(output / FRAGMENT_VALID_FILE, sep="\t", index=False)
    return {"psm_rows": sum(len(item) for item in psms), "fragment_rows": offset}


def write_readme(
    path: Path, build: dict[str, Any], reports: list[dict[str, Any]], has_ms2: bool,
    normalization: dict[str, Any], rt_preparation: dict[str, Any],
) -> None:
    sources = "\n".join(
        f"- `{item['id']}` — {item['rt_rows']:,} RT rows; rt_max={item['rt_max_minutes']:.4f} min"
        for item in reports
    )
    path.write_text(f"""# Fine-tuning dataset `{build['id']}`

This immutable build was assembled from registered Carafe outputs. The sources remain untouched; their checksums and metadata are in `build_manifest.json`.

## Sources

{sources}

## Use

- **Carafe2 RT or all-model fine-tuning:** use `carafe2_input/` as `--in_dir`.
- **DIA-NN RT fine-tuning:** use `diann_rt_input/` as `--in-dir`; it contains the exact same `rt_train_data.tsv` as `carafe2_input/`.

RT normalization policy: `{normalization.get('policy', 'strict_same_global_rt_max')}`. {'Every observed physical `rt` is divided by the common global maximum of ' + f"{float(normalization['target_rt_max_minutes']):.9f}" + ' minutes; no cross-source alignment is fitted. All source observations are retained in `rt_training_observations.tsv`, then repeated peptidoforms are reduced to one median `rt_norm` target in `rt_train_data.tsv`. The aggregation map is `rt_training_aggregation.tsv`.' if normalization.get('policy') == 'rebase_to_global_rt_max' else ('The selected RT target method is `' + str(normalization.get('method')) + '`. All source observations are retained in `rt_training_observations.tsv`; after alignment, repeated peptidoforms are reduced to one median `rt_norm` target in `rt_train_data.tsv`. The aggregation map is `rt_training_aggregation.tsv`.' if normalization.get('policy') == 'aligned' else 'The build fails if source global rt_max values differ beyond the configured tolerance, preventing incomparable gradients from silently being mixed.')} {'MS2 PSM and fragment matrices were re-indexed and merged safely.' if has_ms2 else 'This is an RT-only build; no MS2 files were requested.'}

## RT preparation

```json
{json.dumps(rt_preparation, indent=2)}
```
""", encoding="utf-8")


def build_dataset(catalog_path: Path, build_id: str, output_root_override: Path | None) -> Path:
    catalog = load_catalog(catalog_path)
    build, sources = selected_sources(catalog, build_id)
    include_ms2 = bool(build.get("include_ms2", False))
    reports = [validate_source(source, require_ms2=include_ms2) for source in sources]
    normalization = {**catalog.get("rt_normalization", {}), **build.get("rt_normalization", {})}
    ensure_compatible(reports, normalization=normalization, require_ms2=include_ms2)
    root = output_root_override or Path(catalog.get("output_root", "builds"))
    target = root / build_id
    if target.exists():
        raise FileExistsError(f"Build already exists and is immutable: {target}; define a new build id")
    carafe_input = target / "carafe2_input"
    diann_input = target / "diann_rt_input"
    carafe_input.mkdir(parents=True)
    rt, _, rt_preparation = copy_rt(
        sources, carafe_input, normalization=normalization, reports=reports,
        trypticity=build.get("trypticity"),
    )
    if include_ms2:
        ms2_counts = copy_ms2(sources, carafe_input)
    else:
        ms2_counts = {}
    diann_input.mkdir()
    copy2(carafe_input / RT_FILE, diann_input / RT_FILE)
    copy2(carafe_input / "rt_training_provenance.tsv", diann_input / "rt_training_provenance.tsv")
    manifest = {
        "build_id": build_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog": str(catalog_path.resolve()), "build": build,
        "rt_normalization": normalization, "rt_preparation": rt_preparation, "sources": reports,
        "output": {"rt_rows": int(len(rt)), "rt_unique_peptidoforms": int(rt[["sequence", "mods", "mod_sites"]].drop_duplicates().shape[0]), **ms2_counts},
    }
    (target / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_readme(target / "README.md", build, reports, include_ms2, normalization, rt_preparation)
    return target


def ensure_dataset(catalog_path: Path, build_id: str, output_root_override: Path | None) -> Path:
    """Return an immutable build, creating it only when it does not exist."""
    catalog = load_catalog(catalog_path)
    selected_sources(catalog, build_id)  # Fail early for unknown IDs.
    root = output_root_override or Path(catalog.get("output_root", "builds"))
    target = root / build_id
    if target.is_dir() and (target / "build_manifest.json").is_file():
        return target
    if target.exists():
        raise FileExistsError(f"Build path exists but is incomplete: {target}")
    return build_dataset(catalog_path, build_id, output_root_override)


def main() -> None:
    parser = argparse.ArgumentParser(description="Catalog and build shared Carafe2/DIA-NN fine-tuning datasets")
    parser.add_argument("--catalog", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List registered datasets and builds")
    discover = sub.add_parser("discover-job", help="Print a dataset-entry template from a prepared Carafe job")
    discover.add_argument("--job-dir", type=Path, required=True)
    validate = sub.add_parser("validate", help="Validate sources of one build")
    validate.add_argument("--build", required=True)
    build = sub.add_parser("build", help="Build a combined, model-ready dataset")
    build.add_argument("--build", required=True)
    build.add_argument("--output-root", type=Path)
    ensure = sub.add_parser("ensure", help="Resolve a build ID, creating it if absent")
    ensure.add_argument("--build", required=True)
    ensure.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    catalog = load_catalog(args.catalog)
    if args.command == "discover-job":
        print(yaml.safe_dump([discover_job(args.job_dir)], sort_keys=False, allow_unicode=True))
        return
    if args.command == "list":
        print("Datasets:")
        for item in catalog["datasets"]:
            print(f"  {item['id']}: {item.get('original_experiment_name', '')} ({item.get('organism', 'unknown organism')})")
        print("Builds:")
        for item in catalog["builds"]:
            print(f"  {item['id']}: {', '.join(item.get('datasets', []))}")
        return
    build_config, sources = selected_sources(catalog, args.build)
    if args.command == "validate":
        reports = [validate_source(source, require_ms2=bool(build_config.get("include_ms2", False))) for source in sources]
        normalization = {**catalog.get("rt_normalization", {}), **build_config.get("rt_normalization", {})}
        ensure_compatible(reports, normalization=normalization, require_ms2=bool(build_config.get("include_ms2", False)))
        print(json.dumps(reports, indent=2))
        return
    target = ensure_dataset(args.catalog, args.build, args.output_root) if args.command == "ensure" else build_dataset(args.catalog, args.build, args.output_root)
    print(f"Built shared fine-tuning dataset: {target}")


if __name__ == "__main__":
    main()
