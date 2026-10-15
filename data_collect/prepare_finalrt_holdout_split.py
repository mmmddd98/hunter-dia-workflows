#!/usr/bin/env python3
"""Create one leak-free, source-aware test holdout for final four-source RT builds.

The final RT tables contain one median target per peptidoform.  Their
``rt_training_provenance.tsv`` files retain the individual source observations.
This tool partitions at the peptidoform grain and writes both model-ready
training data and source-/organism-specific test tables without assigning a
multi-source peptidoform to an arbitrary single organism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


KEYS = ["sequence", "mods", "mod_sites"]
HEAVY_NTERM = "Dimethyl:2H(4)13C(2)@N-term"
HEAVY_LYSINE = "Dimethyl:2H(4)13C(2)@K"
DEFAULT_BUILDS = (
    "finalrt_4source_humanref_anchor_v1",
    "finalrt_4source_chronologer_hi_v1",
    "finalrt_4source_humanref_linear_v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=Path("trainingdata"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build", dest="builds", action="append", default=None,
                        help="Final build ID; repeat to override the three default builds")
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=5102026)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_build(root: Path, build_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_dir = root / build_id / "diann_rt_input"
    data_path = input_dir / "rt_train_data.tsv"
    provenance_path = input_dir / "rt_training_provenance.tsv"
    if not data_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(f"{build_id}: expected {data_path} and {provenance_path}")
    data = pd.read_csv(data_path, sep="\t", dtype={"mods": str, "mod_sites": str})
    provenance = pd.read_csv(provenance_path, sep="\t", dtype={"mods": str, "mod_sites": str})
    for frame in (data, provenance):
        frame["mods"] = frame["mods"].fillna("").astype(str)
        frame["mod_sites"] = frame["mod_sites"].fillna("").astype(str)
    required = {"rt_train_row_index", "source_dataset_id", "source_organism"}
    missing = required - set(provenance.columns)
    if missing:
        raise ValueError(f"{provenance_path}: missing {sorted(missing)}")
    if len(data) != provenance["rt_train_row_index"].nunique():
        raise ValueError(f"{build_id}: provenance does not cover each model row once")
    return data, provenance


def source_annotations(data: pd.DataFrame, provenance: pd.DataFrame) -> pd.DataFrame:
    grouped = provenance.groupby("rt_train_row_index", sort=False)
    result = grouped.agg(
        source_dataset_ids=("source_dataset_id", lambda values: ";".join(sorted(set(values.astype(str))))),
        source_organisms=("source_organism", lambda values: ";".join(sorted(set(values.astype(str))))),
        source_dataset_count=("source_dataset_id", "nunique"),
        source_organism_count=("source_organism", "nunique"),
        source_observation_count=("source_dataset_id", "size"),
    ).reindex(range(len(data)))
    if result.isna().any().any():
        raise ValueError("Provenance has missing model-row indices")
    result = result.reset_index(names="rt_train_row_index")

    mods = data["mods"].fillna("").astype(str)
    nterm = mods.str.contains(HEAVY_NTERM, regex=False)
    lysine = mods.str.contains(HEAVY_LYSINE, regex=False)
    heavy = pd.Series("none", index=data.index, dtype="object")
    heavy.loc[nterm & ~lysine] = "heavy_nterm_only"
    heavy.loc[~nterm & lysine] = "heavy_lysine_only"
    heavy.loc[nterm & lysine] = "heavy_both"
    result["heavy_dimethyl_class"] = heavy.to_numpy()
    result["digestion_class"] = data["digestion_class"].fillna("missing").astype(str).to_numpy()
    result["stratum"] = (
        result["source_dataset_ids"] + "|" + result["heavy_dimethyl_class"] + "|" + result["digestion_class"]
    )
    return result


def allocate_test_rows(annotations: pd.DataFrame, fraction: float, seed: int) -> pd.Series:
    """Select exactly round(N*fraction) rows, proportionally within each stratum.

    Largest-remainder allocation avoids a systematic shortfall from independently
    rounding small strata.  Singleton strata normally remain in training, so a
    rare peptidoform is never the sole representative of its stratum in test.
    """
    n_total = len(annotations)
    n_target = round(n_total * fraction)
    counts = annotations.groupby("stratum", sort=True).size().rename("count")
    quotas = counts * fraction
    allocation = quotas.map(math.floor).astype(int)
    remaining = n_target - int(allocation.sum())
    order = sorted(
        counts.index,
        key=lambda label: (-(quotas[label] - allocation[label]), stable_hash(f"{seed}:{label}")),
    )
    for label in order:
        if remaining == 0:
            break
        if allocation[label] < counts[label]:
            allocation[label] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("Could not allocate requested test rows")

    selected: list[int] = []
    for label, group in annotations.groupby("stratum", sort=True):
        ranked = sorted(
            group["rt_train_row_index"].tolist(),
            key=lambda index: stable_hash(f"{seed}:{label}:{index}"),
        )
        selected.extend(ranked[: int(allocation[label])])
    result = pd.Series(False, index=annotations["rt_train_row_index"], dtype=bool)
    result.loc[selected] = True
    if int(result.sum()) != n_target:
        raise RuntimeError("Incorrect test allocation size")
    return result.sort_index()


def add_annotations(data: pd.DataFrame, annotations: pd.DataFrame, split: str) -> pd.DataFrame:
    attached = data.copy()
    attached.insert(0, "rt_train_row_index", range(len(attached)))
    attached = attached.merge(
        annotations.drop(columns=["stratum", "split"]), on="rt_train_row_index", how="left", validate="one_to_one"
    )
    attached["dataset_split"] = split
    if attached["source_dataset_ids"].isna().any():
        raise RuntimeError("Failed to attach source annotations")
    return attached


def source_test_tables(provenance: pd.DataFrame, annotations: pd.DataFrame, output: Path) -> None:
    test_indices = set(annotations.loc[annotations["split"].eq("test"), "rt_train_row_index"])
    observed = provenance[provenance["rt_train_row_index"].isin(test_indices)].copy()
    if observed.empty:
        raise RuntimeError("No test observations")
    (output / "sources").mkdir(parents=True, exist_ok=True)
    (output / "organisms").mkdir(parents=True, exist_ok=True)
    for source_id, frame in observed.groupby("source_dataset_id", sort=True):
        frame.to_csv(output / "sources" / f"{source_id}.tsv", sep="\t", index=False)
    for organism, frame in observed.groupby("source_organism", sort=True):
        frame.to_csv(output / "organisms" / f"{organism}.tsv", sep="\t", index=False)


def summarize(annotations: pd.DataFrame, provenance: pd.DataFrame) -> dict:
    summary: dict[str, object] = {"model_rows": {}}
    for split, frame in annotations.groupby("split", sort=False):
        summary["model_rows"][split] = {
            "rows": int(len(frame)),
            "heavy_dimethyl_class": {str(k): int(v) for k, v in frame["heavy_dimethyl_class"].value_counts().items()},
            "digestion_class": {str(k): int(v) for k, v in frame["digestion_class"].value_counts().items()},
        }
    joined = provenance.merge(annotations[["rt_train_row_index", "split"]], on="rt_train_row_index", how="left")
    summary["source_observations"] = {
        split: {str(k): int(v) for k, v in frame["source_dataset_id"].value_counts().items()}
        for split, frame in joined.groupby("split", sort=False)
    }
    summary["organism_observations"] = {
        split: {str(k): int(v) for k, v in frame["source_organism"].value_counts().items()}
        for split, frame in joined.groupby("split", sort=False)
    }
    return summary


def write_readme(output: Path, manifest: dict) -> None:
    summary = manifest["summary"]
    train = summary["model_rows"]["train"]
    test = summary["model_rows"]["test"]
    source_test = summary["source_observations"]["test"]
    organism_test = summary["organism_observations"]["test"]
    build_lines = "\n".join(f"- `{build}`" for build in manifest["builds"])
    source_lines = "\n".join(f"- `{source}`: {rows:,} Testbeobachtungen" for source, rows in source_test.items())
    organism_lines = "\n".join(f"- `{organism}`: {rows:,} Testbeobachtungen" for organism, rows in organism_test.items())
    output.joinpath("README.md").write_text(
        f"""# Fester Testsplit für die finalen Vier-Quellen-RT-Datensätze

Dieser Ordner definiert den **einmalig festgelegten, unangetasteten Testsatz** für
die folgenden RT-Fine-Tuning-Builds:

{build_lines}

Er dient zur fairen Auswahl und abschließenden Bewertung von DIA-NN-RT-Modellen.
Hyperparameter-Sweeps dürfen ausschließlich die Dateien unter `train/` verwenden.
Die Dateien unter `test/` werden erst nach der Modellauswahl für die finale
Evaluation verwendet.

## Split-Design

- **Grain / Leakage-Schutz:** Der Split erfolgt gruppiert auf
  `sequence + mods + mod_sites` (Peptidoform). Dieselbe Peptidoform kann daher
  niemals im Training und im Test erscheinen, auch nicht über verschiedene
  Quellen hinweg.
- **Testanteil:** exakt {manifest['test_fraction']:.0%} der {train['rows'] + test['rows']:,}
  aggregierten Peptidoformen, also {test['rows']:,} Test- und {train['rows']:,}
  Trainingszeilen.
- **Stratifizierung:** Die Ziehung bewahrt die Verteilung der Kombination aus
  Quell-Datensatz, Heavy-Dimethyl-Klasse und Verdauklasse (`digestion_class`).
  Es werden keine Zeilen verworfen oder nach dem Outcome ausgewählt.
- **Mehrquellen-Peptidoformen:** Eine Peptidoform erhält genau eine Split-Zuordnung.
  Ihre vollständige Herkunft bleibt als `source_dataset_ids` und
  `source_organisms` erhalten. Eine einzelne `source_organism`-Angabe wäre für
  solche Zeilen falsch.
- **Reproduzierbarkeit:** deterministische Hash-Ziehung, Seed
  `{manifest['seed']}`. Eingabe-Prüfsummen und die vollständige Zuordnung stehen
  in `split_manifest.json` bzw. `split_annotations.tsv`.

## Zusammensetzung

### Heavy-Dimethyl-Klassen

| Klasse | Training | Test |
| --- | ---: | ---: |
{chr(10).join(f"| {name} | {count:,} | {test['heavy_dimethyl_class'].get(name, 0):,} |" for name, count in train['heavy_dimethyl_class'].items())}

### Verdauklassen

| Klasse | Training | Test |
| --- | ---: | ---: |
{chr(10).join(f"| {name} | {count:,} | {test['digestion_class'].get(name, 0):,} |" for name, count in train['digestion_class'].items())}

### Testbeobachtungen nach Quelle

{source_lines}

### Testbeobachtungen nach Organismusgruppe

{organism_lines}

## Dateistruktur und Verwendung

Für jeden Build existieren:

```text
<build>/train/diann_rt_input/rt_train_data.tsv
<build>/test/diann_rt_input/rt_test_data.tsv
<build>/test/sources/<source_dataset_id>.tsv
<build>/test/organisms/{{celegans,human,triplemix}}.tsv
```

Die Trainings- und aggregierten Testtabellen enthalten zusätzlich zu den RT- und
Tryptizitätsfeldern die Herkunftsfelder `source_dataset_ids`, `source_organisms`,
`source_dataset_count`, `source_organism_count`, `source_observation_count` und
`dataset_split`. Für quellen- oder organismusspezifische Metriken verwende die
Tabellen unter `test/sources/` bzw. `test/organisms/`: Sie behalten je Quelle die
beobachtete, ausgerichtete RT und ihre exakte Provenance bei.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if not 0 < args.test_fraction < 0.5:
        raise SystemExit("--test-fraction must be between 0 and 0.5")
    builds = tuple(args.builds or DEFAULT_BUILDS)
    output = args.output_dir.expanduser().resolve()
    root = args.training_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    canonical_data, canonical_provenance = read_build(root, builds[0])
    canonical_keys = canonical_data[KEYS].copy()
    for build_id in builds[1:]:
        data, provenance = read_build(root, build_id)
        if not data[KEYS].equals(canonical_keys):
            raise ValueError(f"{build_id}: peptidoform rows differ from {builds[0]}")
        if not provenance[["rt_train_row_index", "source_dataset_id", "source_organism"]].equals(
            canonical_provenance[["rt_train_row_index", "source_dataset_id", "source_organism"]]
        ):
            raise ValueError(f"{build_id}: source provenance differs from {builds[0]}")

    annotations = source_annotations(canonical_data, canonical_provenance)
    is_test = allocate_test_rows(annotations, args.test_fraction, args.seed)
    annotations["split"] = annotations["rt_train_row_index"].map(is_test).map({True: "test", False: "train"})
    if annotations["split"].isna().any():
        raise RuntimeError("Split assignment missing")

    output.mkdir(parents=True)
    annotations.to_csv(output / "split_annotations.tsv", sep="\t", index=False)
    for build_id in builds:
        data, provenance = read_build(root, build_id)
        build_root = output / build_id
        model_with_annotations = add_annotations(data, annotations, "all")
        train = model_with_annotations[annotations["split"].eq("train").to_numpy()].copy()
        test = model_with_annotations[annotations["split"].eq("test").to_numpy()].copy()
        train["dataset_split"] = "train"
        test["dataset_split"] = "test"
        train_path = build_root / "train" / "diann_rt_input" / "rt_train_data.tsv"
        test_path = build_root / "test" / "diann_rt_input" / "rt_test_data.tsv"
        train_path.parent.mkdir(parents=True)
        test_path.parent.mkdir(parents=True)
        train.to_csv(train_path, sep="\t", index=False)
        test.to_csv(test_path, sep="\t", index=False)
        source_test_tables(provenance, annotations, build_root / "test")

    manifest = {
        "schema_version": 1,
        "purpose": "fixed grouped test holdout for final four-source RT fine-tuning sweeps",
        "builds": list(builds),
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "grouping_key": KEYS,
        "stratification": ["source_dataset_ids", "heavy_dimethyl_class", "digestion_class"],
        "multi_source_policy": (
            "A peptidoform is assigned once. Source/organism memberships are retained as semicolon-separated sets; "
            "source-specific test tables retain one row per source observation."
        ),
        "inputs": {
            build_id: {
                "rt_train_data_sha256": sha256(root / build_id / "diann_rt_input" / "rt_train_data.tsv"),
                "provenance_sha256": sha256(root / build_id / "diann_rt_input" / "rt_training_provenance.tsv"),
            }
            for build_id in builds
        },
        "summary": summarize(annotations, canonical_provenance),
    }
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_readme(output, manifest)
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
