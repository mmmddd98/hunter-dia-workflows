#!/usr/bin/env python3
"""Create a versioned, FASTA-backed trypticity overlay for RT training data.

The immutable source builds are never modified.  Instead, this script copies
them below ``trainingdata/rt_trypticity_annotated_v1/datasets`` and appends
strict per-row digestion annotations to both DIA-NN and Carafe2 RT inputs.

Classification is based on every DIA-NN-reported protein accession and every
matching occurrence in the exact search FASTA.  Conflicting protein contexts
remain ``ambiguous``; absent mappings remain ``unmapped``.  No sequence-only
guessing is used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

import pandas as pd


ANNOTATION_VERSION = "rt_trypticity_annotated_v1"
EXPECTED_CUT = "R*"
EXPECTED_SEMI = "1"
ANNOTATION_COLUMNS = (
    "digestion_class",
    "trypticity_detail",
    "is_semi_tryptic",
    "enzymatic_n_terminus",
    "enzymatic_c_terminus",
    "digestion_rule",
    "digestion_mapping_count",
    "digestion_mapped_protein_count",
    "digestion_reported_protein_count",
    "digestion_mapping_classes",
)


@dataclass(frozen=True)
class SourceSpec:
    organism: str
    build_id: str
    report: Path
    fasta: Path
    digestion_contract: Path


def default_specs() -> tuple[SourceSpec, ...]:
    share = Path("./shared_data")
    carafe = share / "carafe_share/2_5_0"
    return (
        SourceSpec(
            organism="celegans",
            build_id="celegans_exploris_44min_rtms2_v1",
            report=carafe / "test_carafe_step2_lib_prod/carafe/balanced/carafe_pipeline_test/input/diann_step2_report.parquet",
            fasta=share / "fastas/uniprotkb_proteome_UP000001940_2026_04_10.fasta",
            digestion_contract=carafe / "test_carafe_step2_lib_prod/carafe/balanced/carafe_pipeline_test/meta/digestion_contract.env",
        ),
        SourceSpec(
            organism="triplemix",
            build_id="triplemix_source_a_entrap_neu_filtered_v1",
            report=carafe / "triplemix_source_a_entrap_neu/carafe/balanced/source_a_triple_cluster_filtered/input/diann_step2_report.parquet",
            fasta=share / "fastas/homo_sapiens_Arabidopsis_thaliana_escherichia_coli_source_a.fasta",
            digestion_contract=carafe / "triplemix_source_a_entrap_neu/carafe/balanced/source_a_triple_cluster_filtered/meta/digestion_contract.env",
        ),
        SourceSpec(
            organism="human",
            build_id="human_source_hunter_human_rtms2_v1",
            report=carafe / "carafe_trainingdateta_human_source_human/carafe/balanced/human_source_hunter_human/input/diann_step2_report.parquet",
            fasta=share / "fastas/UniProt_Proteome_Human_2026_01.fasta",
            digestion_contract=carafe / "carafe_trainingdateta_human_source_human/carafe/balanced/human_source_hunter_human/meta/digestion_contract.env",
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "trainingdata",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=f"Defaults to TRAINING_ROOT/{ANNOTATION_VERSION}",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_contract(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key] = value.replace("\\*", "*").strip("'\"")
    return values


def validate_spec(spec: SourceSpec, training_root: Path) -> dict[str, str]:
    paths = {
        "build": training_root / spec.build_id,
        "report": spec.report,
        "fasta": spec.fasta,
        "digestion_contract": spec.digestion_contract,
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source inputs:\n  " + "\n  ".join(missing))
    contract = parse_contract(spec.digestion_contract)
    if contract.get("DIANN_CUT") != EXPECTED_CUT:
        raise ValueError(
            f"{spec.organism}: expected DIANN_CUT={EXPECTED_CUT}, "
            f"found {contract.get('DIANN_CUT')!r}"
        )
    if contract.get("DIANN_SEMI") != EXPECTED_SEMI:
        raise ValueError(
            f"{spec.organism}: expected semi-specific search, "
            f"found DIANN_SEMI={contract.get('DIANN_SEMI')!r}"
        )
    return contract


def fasta_accession(header: str) -> str:
    token = header[1:].strip().split(maxsplit=1)[0]
    if token.count("|") >= 2:
        return token.split("|")[1]
    return token


def load_fasta(path: Path) -> dict[str, tuple[str, ...]]:
    records: dict[str, list[str]] = defaultdict(list)
    header: str | None = None
    sequence_parts: list[str] = []

    def store() -> None:
        if header is None:
            return
        sequence = "".join(sequence_parts).upper()
        if sequence and sequence not in records[fasta_accession(header)]:
            records[fasta_accession(header)].append(sequence)

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith(">"):
                store()
                header = stripped
                sequence_parts = []
            elif header is not None:
                sequence_parts.append(stripped)
        store()
    if not records:
        raise ValueError(f"No FASTA records parsed from {path}")
    return {accession: tuple(sequences) for accession, sequences in records.items()}


def report_protein_mapping(path: Path) -> dict[str, tuple[str, ...]]:
    report = pd.read_parquet(path, columns=["Stripped.Sequence", "Protein.Ids"])
    report = report.dropna().drop_duplicates()
    mapping: dict[str, set[str]] = defaultdict(set)
    for sequence, protein_ids in report.itertuples(index=False, name=None):
        for accession in str(protein_ids).split(";"):
            cleaned = accession.strip().split(maxsplit=1)[0]
            if cleaned:
                mapping[str(sequence).upper()].add(cleaned)
    return {sequence: tuple(sorted(accessions)) for sequence, accessions in mapping.items()}


def peptide_occurrences(peptide: str, protein: str) -> Iterable[tuple[int, int]]:
    start = 0
    while True:
        position = protein.find(peptide, start)
        if position < 0:
            return
        yield position, position + len(peptide)
        start = position + 1


def occurrence_class(peptide: str, protein: str, start: int, end: int) -> str:
    n_enzymatic = start == 0 or protein[start - 1] == "R"
    c_enzymatic = end == len(protein) or peptide.endswith("R")
    if n_enzymatic and c_enzymatic:
        return "fully_tryptic"
    if not n_enzymatic and c_enzymatic:
        return "semi_nterm"
    if n_enzymatic and not c_enzymatic:
        return "semi_cterm"
    return "non_specific"


def summarize_mapping_classes(classes: list[str]) -> dict[str, object]:
    unique = tuple(sorted(set(classes)))
    if not classes:
        detail = "unmapped"
    elif len(unique) == 1:
        detail = unique[0]
    else:
        detail = "ambiguous"

    if detail == "fully_tryptic":
        digestion_class = "fully_tryptic"
        is_semi: object = False
        n_enzymatic: object = True
        c_enzymatic: object = True
    elif detail == "semi_nterm":
        digestion_class = "semi_tryptic"
        is_semi = True
        n_enzymatic = False
        c_enzymatic = True
    elif detail == "semi_cterm":
        digestion_class = "semi_tryptic"
        is_semi = True
        n_enzymatic = True
        c_enzymatic = False
    elif detail == "non_specific":
        digestion_class = "non_specific"
        is_semi = pd.NA
        n_enzymatic = False
        c_enzymatic = False
    else:
        digestion_class = detail
        is_semi = pd.NA
        n_enzymatic = pd.NA
        c_enzymatic = pd.NA
    return {
        "digestion_class": digestion_class,
        "trypticity_detail": detail,
        "is_semi_tryptic": is_semi,
        "enzymatic_n_terminus": n_enzymatic,
        "enzymatic_c_terminus": c_enzymatic,
        "digestion_mapping_classes": ";".join(unique),
    }


def annotate_sequences(
    sequences: Iterable[str],
    protein_mapping: dict[str, tuple[str, ...]],
    fasta: dict[str, tuple[str, ...]],
    organism: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sequence_value in sorted(set(sequences)):
        sequence = str(sequence_value).upper()
        reported = protein_mapping.get(sequence, ())
        classes: list[str] = []
        mapped_proteins: set[str] = set()
        for accession in reported:
            for protein in fasta.get(accession, ()):
                occurrences = list(peptide_occurrences(sequence, protein))
                if occurrences:
                    mapped_proteins.add(accession)
                classes.extend(
                    occurrence_class(sequence, protein, start, end)
                    for start, end in occurrences
                )
        summary = summarize_mapping_classes(classes)
        rows.append(
            {
                "source_organism": organism,
                "sequence": sequence,
                **summary,
                "digestion_rule": EXPECTED_CUT,
                "digestion_mapping_count": len(classes),
                "digestion_mapped_protein_count": len(mapped_proteins),
                "digestion_reported_protein_count": len(reported),
            }
        )
    return pd.DataFrame(rows)


def attach_annotations(frame: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    overlap = set(ANNOTATION_COLUMNS) & set(frame.columns)
    if overlap:
        raise ValueError(f"Input training table already has annotation columns: {sorted(overlap)}")
    before = len(frame)
    out = frame.merge(
        annotations.drop(columns=["source_organism"]),
        on="sequence",
        how="left",
        validate="many_to_one",
    )
    if len(out) != before:
        raise RuntimeError("Annotation join changed training row count")
    if out["digestion_class"].isna().any():
        missing = int(out["digestion_class"].isna().sum())
        raise RuntimeError(f"{missing} training rows lack trypticity annotations")
    return out


def build_source(
    spec: SourceSpec,
    training_root: Path,
    destination_root: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    contract = validate_spec(spec, training_root)
    source_build = training_root / spec.build_id
    destination_build = destination_root / "datasets" / spec.build_id
    shutil.copytree(source_build, destination_build)

    rt_path = source_build / "diann_rt_input/rt_train_data.tsv"
    frame = pd.read_csv(rt_path, sep="\t", dtype={"mods": str, "mod_sites": str})
    proteins = report_protein_mapping(spec.report)
    fasta = load_fasta(spec.fasta)
    annotations = annotate_sequences(frame.sequence.astype(str), proteins, fasta, spec.organism)
    annotated = attach_annotations(frame, annotations)

    for relative in (
        Path("diann_rt_input/rt_train_data.tsv"),
        Path("carafe2_input/rt_train_data.tsv"),
    ):
        annotated.to_csv(destination_build / relative, sep="\t", index=False)

    source_manifest_path = source_build / "build_manifest.json"
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    manifest["annotation_overlay"] = {
        "version": ANNOTATION_VERSION,
        "source_build": str(source_build),
        "source_build_manifest_sha256": sha256(source_manifest_path),
        "diann_report": str(spec.report),
        "diann_report_sha256": sha256(spec.report),
        "fasta": str(spec.fasta),
        "fasta_sha256": sha256(spec.fasta),
        "digestion_contract": str(spec.digestion_contract),
        "digestion_contract_sha256": sha256(spec.digestion_contract),
        "digestion_rule": contract["DIANN_CUT"],
        "semi_specific_search": contract["DIANN_SEMI"] == "1",
        "missed_cleavages": int(contract["DIANN_MISSED_CLEAVAGES"]),
        "classification_policy": (
            "all DIA-NN-reported proteins and all matching FASTA occurrences; "
            "conflicting contexts are ambiguous"
        ),
        "annotation_columns": list(ANNOTATION_COLUMNS),
        "row_count": len(annotated),
        "sequence_count": int(annotated.sequence.nunique()),
        "class_counts_rows": {
            str(key): int(value)
            for key, value in annotated.digestion_class.value_counts().sort_index().items()
        },
        "detail_counts_sequences": {
            str(key): int(value)
            for key, value in annotations.trypticity_detail.value_counts().sort_index().items()
        },
    }
    (destination_build / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    readme = (destination_build / "README.md").read_text(encoding="utf-8")
    readme += (
        "\n## Trypticity annotation overlay\n\n"
        f"This is a versioned derivative created by `{ANNOTATION_VERSION}`. "
        "The original immutable build remains unchanged. Per-row digestion classes "
        "were reconstructed from the original DIA-NN protein assignments, exact search "
        "FASTA and the recorded `R*` semi-specific digestion contract. Conflicting "
        "protein contexts are marked `ambiguous`. See `build_manifest.json` for hashes "
        "and coverage.\n"
    )
    (destination_build / "README.md").write_text(readme, encoding="utf-8")

    quality = {
        "organism": spec.organism,
        "build_id": spec.build_id,
        "rows": len(annotated),
        "unique_sequences": int(annotated.sequence.nunique()),
        "report_sequence_coverage": float(annotations.digestion_reported_protein_count.gt(0).mean()),
        "fasta_mapping_coverage": float(annotations.digestion_mapping_count.gt(0).mean()),
        "digestion_class_rows": {
            str(key): int(value)
            for key, value in annotated.digestion_class.value_counts().sort_index().items()
        },
        "trypticity_detail_sequences": {
            str(key): int(value)
            for key, value in annotations.trypticity_detail.value_counts().sort_index().items()
        },
        "ambiguous_sequence_fraction": float(annotations.trypticity_detail.eq("ambiguous").mean()),
        "unmapped_sequence_fraction": float(annotations.trypticity_detail.eq("unmapped").mean()),
    }
    return annotations, quality


def main() -> None:
    args = parse_args()
    training_root = args.training_root.resolve()
    output_root = (args.output_root or training_root / ANNOTATION_VERSION).resolve()
    if output_root.exists():
        raise SystemExit(
            f"Refusing to overwrite existing annotation build: {output_root}\n"
            "Choose a new --output-root or remove the incomplete output manually."
        )
    for spec in default_specs():
        validate_spec(spec, training_root)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}.", dir=output_root.parent
    ) as temporary:
        stage = Path(temporary) / output_root.name
        stage.mkdir()
        annotation_frames = []
        quality_sources = []
        for spec in default_specs():
            annotations, quality = build_source(spec, training_root, stage)
            annotation_frames.append(annotations)
            quality_sources.append(quality)
        catalog = pd.concat(annotation_frames, ignore_index=True)
        if catalog.duplicated(["source_organism", "sequence"]).any():
            raise RuntimeError("Annotation catalog has duplicate organism/sequence keys")
        catalog.to_csv(stage / "rt_trypticity_annotations.tsv", sep="\t", index=False)
        quality_report = {
            "annotation_version": ANNOTATION_VERSION,
            "digestion_rule": EXPECTED_CUT,
            "classification_grain": "source_organism + unmodified peptide sequence",
            "sources": quality_sources,
            "catalog_rows": len(catalog),
            "catalog_sha256": sha256(stage / "rt_trypticity_annotations.tsv"),
        }
        (stage / "quality_report.json").write_text(
            json.dumps(quality_report, indent=2) + "\n", encoding="utf-8"
        )
        stage.rename(output_root)

    print(f"Created trypticity-annotated training-data overlay: {output_root}")
    print(f"Catalog: {output_root / 'rt_trypticity_annotations.tsv'}")


if __name__ == "__main__":
    main()
