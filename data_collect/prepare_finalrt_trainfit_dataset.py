#!/usr/bin/env python3
"""Create train-only RT-aligned train/validation/test inputs for DIA-NN sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prepare_finalrt_holdout_split import (
    DEFAULT_BUILDS,
    KEYS,
    allocate_test_rows,
    read_build,
    sha256,
)
from rt_alignment import apply_rt_target, fit_target_alignment, fit_target_inverse


CANONICAL_BUILD = "finalrt_4source_humanref_anchor_v1"
MODES = ("anchor_irt", "chronologer_hi", "linear", "global_60min")
REFERENCE_SOURCE = "human_source_hunter_human_v1"
REFERENCE_MINUTES = 60.101620738856


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=Path("trainingdata"))
    parser.add_argument(
        "--holdout-dir", type=Path,
        default=Path("trainingdata/finalrt_4source_holdout15_stratified_v1"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.10,
                        help="Fraction of the non-test development set")
    parser.add_argument("--validation-seed", type=int, default=5102027)
    return parser.parse_args()


def read_fixed_annotations(path: Path, n_rows: int) -> pd.DataFrame:
    annotations = pd.read_csv(path / "split_annotations.tsv", sep="\t")
    required = {
        "rt_train_row_index", "source_dataset_ids", "source_organisms",
        "source_dataset_count", "source_organism_count", "source_observation_count",
        "heavy_dimethyl_class", "digestion_class", "stratum", "split",
    }
    missing = required - set(annotations.columns)
    if missing:
        raise ValueError(f"{path}: split annotations missing {sorted(missing)}")
    if len(annotations) != n_rows or not annotations.rt_train_row_index.is_unique:
        raise ValueError("Fixed holdout annotations do not match the canonical model table")
    if set(annotations.split) != {"train", "test"}:
        raise ValueError("Fixed holdout must contain exactly train and test labels")
    return annotations.sort_values("rt_train_row_index").reset_index(drop=True)


def add_validation_split(annotations: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if not 0 < fraction < 0.5:
        raise ValueError("validation fraction must be between 0 and 0.5")
    result = annotations.copy()
    development = result.loc[result.split.eq("train")].copy()
    is_validation = allocate_test_rows(development, fraction, seed)
    validation_indices = set(is_validation[is_validation].index)
    result.loc[result.rt_train_row_index.isin(validation_indices), "split"] = "validation"
    expected = round(len(development) * fraction)
    if int(result.split.eq("validation").sum()) != expected:
        raise RuntimeError("Incorrect validation allocation")
    return result


def observations_with_split(training_root: Path, annotations: pd.DataFrame) -> pd.DataFrame:
    _, provenance = read_build(training_root, CANONICAL_BUILD)
    columns = ["rt_train_row_index", "split"]
    merged = provenance.merge(annotations[columns], on="rt_train_row_index", how="left", validate="many_to_one")
    if merged.split.isna().any():
        raise RuntimeError("An observation lacks a split assignment")
    # The existing rt_norm is from the all-data build and must never influence
    # this derived dataset. apply_rt_target overwrites it after train-only fitting.
    return merged.drop(columns=["rt_norm"])


def frames_by_source(observations: pd.DataFrame, split: str) -> dict[str, pd.DataFrame]:
    subset = observations.loc[observations.split.eq(split)].copy()
    frames = {source: frame.drop(columns=["split"]).copy()
              for source, frame in subset.groupby("source_dataset_id", sort=True)}
    if set(frames) != set(observations.source_dataset_id.unique()):
        raise ValueError(f"{split}: at least one source has no observations")
    return frames


def aggregate_model_table(
    transformed: dict[str, pd.DataFrame], base_data: pd.DataFrame, annotations: pd.DataFrame, split: str,
) -> pd.DataFrame:
    observations = pd.concat(transformed.values(), ignore_index=True)
    model = observations.groupby(KEYS, as_index=False, dropna=False).agg(rt=("rt", "median"), rt_norm=("rt_norm", "median"))
    source_columns = [
        "rt_train_row_index", "source_dataset_ids", "source_organisms", "source_dataset_count",
        "source_organism_count", "source_observation_count", "heavy_dimethyl_class", "digestion_class",
    ]
    base = base_data.copy()
    base.insert(0, "rt_train_row_index", range(len(base)))
    annotation_names = set(source_columns) - {"rt_train_row_index"}
    extra = [
        column for column in base.columns
        if column not in {"rt_train_row_index", *KEYS, "rt", "rt_norm", *annotation_names}
    ]
    metadata = base[["rt_train_row_index", *KEYS, *extra]].merge(
        annotations[source_columns], on="rt_train_row_index", how="left", validate="one_to_one"
    )
    model = model.merge(metadata, on=KEYS, how="left", validate="one_to_one")
    model["dataset_split"] = split
    required = ["rt_train_row_index", "rt", "rt_norm", "source_dataset_ids", "source_organisms"]
    if model[required].isna().any().any():
        missing = model[required].columns[model[required].isna().any()].tolist()
        raise RuntimeError(f"{split}: missing required metadata after aggregation: {missing}")
    return model


def write_observation_tables(transformed: dict[str, pd.DataFrame], directory: Path) -> None:
    observations = pd.concat(transformed.values(), ignore_index=True)
    (directory / "sources").mkdir(parents=True, exist_ok=True)
    (directory / "organisms").mkdir(parents=True, exist_ok=True)
    observations.to_csv(directory / "rt_observations.tsv", sep="\t", index=False)
    for source, frame in observations.groupby("source_dataset_id", sort=True):
        frame.to_csv(directory / "sources" / f"{source}.tsv", sep="\t", index=False)
    for organism, frame in observations.groupby("source_organism", sort=True):
        frame.to_csv(directory / "organisms" / f"{organism}.tsv", sep="\t", index=False)


def distribution(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame[column].value_counts().items()}


def write_readme(output: Path, manifest: dict) -> None:
    counts = manifest["model_rows"]
    output.joinpath("README.md").write_text(
        f"""# Train-only ausgerichtete Vierquellen-RT-Daten für DIA-NN-Sweeps

Dieser Datensatz verwendet den festen Peptidoform-Split aus
`finalrt_4source_holdout15_stratified_v1`, verhindert jedoch zusätzlich
**Alignment-Leakage**: Jede RT-Ausrichtung wird ausschließlich mit den
Quellbeobachtungen der Trainingspartition geschätzt. Die so eingefrorene
Abbildung wird anschließend unverändert auf Training, Validierung und den
finalen Test angewandt.

## Partitionen

| Partition | Aggregierte Peptidoformen | Verwendung |
|---|---:|---|
| Training | {counts['train']:,} | Modellparameter lernen und Alignment fitten |
| Validation | {counts['validation']:,} | Sweep-Auswahl und Early Stopping |
| Test | {counts['test']:,} | einmalige finale Bewertung |

Der finale Test umfasst weiterhin exakt 15 % aller 79.000 Peptidoformen. Die
Validation umfasst {manifest['validation_fraction']:.0%} der verbleibenden
Entwicklungsdaten. Alle Splits sind gruppiert auf
`sequence + mods + mod_sites`; eine Peptidoform und alle ihre Quellmessungen
kommen daher in genau einer Partition vor.

## Zielraum-Varianten

- `anchor_irt`: gemeinsame robuste Anchor-RT-Abbildung auf Human als Referenz.
- `chronologer_hi`: quellspezifische Abbildung auf die externe Chronologer-HI-Skala.
- `linear`: robuste lineare Abbildung auf die Human-Referenz.
- `global_60min`: keine Ausrichtung; Roh-RT geteilt durch 60,101620738856 min.

Die ersten drei Varianten erhalten ihre Alignment-Parameter **nur aus
Training**. Diese Parameter und die jeweilige Rückabbildung in beobachtete
RT-Minuten stehen in `<mode>/rt_alignment.json`. `global_60min` ist die
Negativkontrolle ohne geschätzte Parameter.

## Verwendung

Für jeden Zielraum liegen modellfertige Tabellen unter:

```text
<mode>/train/diann_rt_input/rt_train_data.tsv
<mode>/validation/diann_rt_input/rt_validation_data.tsv
<mode>/test/diann_rt_input/rt_test_data.tsv
```

Die source-/organismusspezifischen Test- und Validationstabellen liegen unter
`<mode>/<partition>/sources/` und `<mode>/<partition>/organisms/`. Sie behalten
pro Quellbeobachtung die beobachtete RT in Minuten, den train-only-transformierten
`rt_norm` und `source_dataset_id` bzw. `source_organism`.

Die Auswahl im Sweep darf nur `train/` und `validation/` verwenden. Der Ordner
`test/` bleibt bis zur finalen Modellauswahl unangetastet.

Vollständige Parameter, Prüfsummen und die Verteilungen nach Quelle,
Heavy-Dimethyl und Tryptizität stehen in `dataset_manifest.json`.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    root, holdout = args.training_root.resolve(), args.holdout_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    base_data, _ = read_build(root, CANONICAL_BUILD)
    annotations = add_validation_split(read_fixed_annotations(holdout, len(base_data)), args.validation_fraction, args.validation_seed)
    observations = observations_with_split(root, annotations)
    train_frames = frames_by_source(observations, "train")

    output.mkdir(parents=True)
    annotations.to_csv(output / "split_annotations.tsv", sep="\t", index=False)
    model_rows: dict[str, int] = {}
    summaries: dict[str, dict] = {}
    for mode in MODES:
        alignment = fit_target_alignment(train_frames, mode=mode, reference_source=REFERENCE_SOURCE)
        inverse = fit_target_inverse(train_frames, mode=mode, reference_minutes=REFERENCE_MINUTES, alignment=alignment)
        mode_root = output / mode
        mode_root.mkdir()
        _, metadata = apply_rt_target(
            train_frames, mode=mode, reference_minutes=REFERENCE_MINUTES,
            alignment=alignment, inverse_coefficients=inverse,
        )
        (mode_root / "rt_alignment.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        summaries[mode] = {"alignment": metadata.get("alignment", {"mode": mode})}
        for split in ("train", "validation", "test"):
            transformed, _ = apply_rt_target(
                frames_by_source(observations, split), mode=mode, reference_minutes=REFERENCE_MINUTES,
                alignment=alignment, inverse_coefficients=inverse,
            )
            model = aggregate_model_table(transformed, base_data, annotations, split)
            target = mode_root / split / "diann_rt_input"
            target.mkdir(parents=True)
            filename = "rt_train_data.tsv" if split == "train" else f"rt_{split}_data.tsv"
            model.to_csv(target / filename, sep="\t", index=False)
            write_observation_tables(transformed, mode_root / split)
            model_rows.setdefault(split, len(model))
            if model_rows[split] != len(model):
                raise RuntimeError(f"{mode}: model-row count differs for {split}")
            summaries[mode][split] = {
                "model_rows": int(len(model)),
                "heavy_dimethyl_class": distribution(model, "heavy_dimethyl_class"),
                "digestion_class": distribution(model, "digestion_class"),
                "source_observations": distribution(pd.concat(transformed.values()), "source_dataset_id"),
            }

    manifest = {
        "schema_version": 1,
        "purpose": "DIA-NN normalization and fine-tuning sweep with train-only target alignment",
        "canonical_raw_observation_build": CANONICAL_BUILD,
        "fixed_test_holdout": str(holdout),
        "fixed_test_manifest_sha256": sha256(holdout / "split_manifest.json"),
        "validation_fraction_of_development": args.validation_fraction,
        "validation_seed": args.validation_seed,
        "grouping_key": KEYS,
        "reference_source": REFERENCE_SOURCE,
        "reference_minutes": REFERENCE_MINUTES,
        "modes": list(MODES),
        "fit_rule": "Alignment and target-to-observed inverse maps are fitted from training peptidoforms only.",
        "model_rows": model_rows,
        "mode_summaries": summaries,
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_readme(output, manifest)
    print(json.dumps({"model_rows": model_rows, "modes": list(MODES)}, indent=2))


if __name__ == "__main__":
    main()
