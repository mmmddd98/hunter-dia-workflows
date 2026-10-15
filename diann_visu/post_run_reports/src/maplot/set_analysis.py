from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Literal

import polars as pl


EntityLevel = Literal["peptide", "peptidoform", "nterm", "protein"]

AA_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWY")
ENTITY_LEVEL_LABELS: dict[EntityLevel, str] = {
    "peptide": "Stripped peptide",
    "peptidoform": "Modified sequence",
    "nterm": "N-terminal peptidoform",
    "protein": "Protein group",
}
ENTITY_LEVEL_DESCRIPTIONS: dict[EntityLevel, str] = {
    "peptide": "unique stripped peptide sequence",
    "peptidoform": "unique DIA-NN Modified.Sequence including modifications",
    "nterm": "unique nterm_modif_peptide: N-terminal modification class plus stripped peptide",
    "protein": "unique protein group identifier",
}
ENTITY_COLUMNS: dict[EntityLevel, tuple[str, ...]] = {
    "peptide": ("Stripped.Sequence", "peptide"),
    "peptidoform": ("Modified.Sequence",),
    "nterm": ("nterm_modif_peptide",),
    "protein": ("Protein.Group", "Protein.Ids", "protein"),
}


@dataclass(frozen=True)
class IntersectionSelection:
    """Exact set membership selected from a multi-search comparison."""

    key: str
    present_searches: tuple[str, ...]


def entity_sets(tables: dict[str, pl.DataFrame], entity_level: EntityLevel) -> dict[str, set[str]]:
    """Return unique entity identifiers per search.

    The function accepts raw DIA-NN tables, adapted peptide matrices, or
    termini-annotated tables. It picks the first suitable identifier column for
    the requested entity level.
    """

    sets: dict[str, set[str]] = {}
    for search, df in tables.items():
        expr = entity_identifier_expr(df, entity_level)
        if expr is None:
            sets[search] = set()
            continue
        sets[search] = set(
            df.select(expr.alias("entity"))
            .drop_nulls()
            .unique()
            .get_column("entity")
            .to_list()
        )
    return sets


def intersection_summary(
    tables: dict[str, pl.DataFrame],
    entity_level: EntityLevel,
    *,
    searches: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Build exact search-membership counts for one entity level."""

    sets = entity_sets(tables, entity_level)
    selected = tuple(searches or sorted(sets))
    if not selected:
        return empty_intersection_frame()

    membership: dict[str, tuple[str, ...]] = {}
    for search in selected:
        for entity in sets.get(search, set()):
            current = set(membership.get(entity, ()))
            current.add(search)
            membership[entity] = tuple(sorted(current))

    counts = Counter(membership.values())
    rows = []
    for present, count in counts.items():
        absent = tuple(search for search in selected if search not in present)
        rows.append(
            {
                "intersection_key": encode_intersection(present),
                "present_searches": " & ".join(present),
                "absent_searches": " & ".join(absent),
                "count": count,
                "n_searches": len(present),
            }
        )
    if not rows:
        return empty_intersection_frame()
    return pl.DataFrame(rows).sort(["count", "present_searches"], descending=[True, False])


def intersection_entities(
    tables: dict[str, pl.DataFrame],
    entity_level: EntityLevel,
    intersection_key: str,
    *,
    searches: Iterable[str] | None = None,
) -> list[str]:
    """Return entities that occur in exactly the selected searches."""

    sets = entity_sets(tables, entity_level)
    selected = tuple(searches or sorted(sets))
    present = set(decode_intersection(intersection_key))
    if not selected or not present:
        return []

    included = [sets.get(search, set()) for search in selected if search in present]
    if not included:
        return []
    entities = set.intersection(*included)
    for search in selected:
        if search not in present:
            entities -= sets.get(search, set())
    return sorted(entities)


def entity_feature_table(
    *,
    raw_tables: dict[str, pl.DataFrame] | None = None,
    adapted_tables: dict[str, pl.DataFrame] | None = None,
    termini_tables: dict[str, pl.DataFrame] | None = None,
    entity_level: EntityLevel,
    entities: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Collect analyzable properties for entities across searches.

    The output is long-form: one row per search/entity where the entity is
    observed. It intentionally keeps only lightweight, broadly available
    features so the same table can power static reports and interactive Dash
    plots.
    """

    entity_filter = set(entities) if entities is not None else None
    rows: list[dict[str, object]] = []
    sources = tables_for_level(raw_tables, adapted_tables, termini_tables, entity_level)
    for search, df in sources.items():
        rows.extend(feature_rows_for_search(search, df, entity_level, entity_filter))
    if not rows:
        return empty_feature_frame()
    return pl.DataFrame(rows)


def add_intersection_labels(
    features: pl.DataFrame,
    source_tables: dict[str, pl.DataFrame],
    entity_level: EntityLevel,
) -> pl.DataFrame:
    """Attach exact membership labels to a feature table."""

    if features.is_empty() or "entity" not in features.columns:
        return features
    sets = entity_sets(source_tables, entity_level)
    labels = []
    for entity in features.get_column("entity").to_list():
        present = tuple(search for search in sorted(sets) if entity in sets[search])
        labels.append(encode_intersection(present))
    return features.with_columns(pl.Series("intersection_key", labels))


def encode_intersection(searches: Iterable[str]) -> str:
    return "||".join(sorted(searches))


def decode_intersection(key: str | None) -> tuple[str, ...]:
    if not key:
        return ()
    return tuple(part for part in key.split("||") if part)


def feature_rows_for_search(
    search: str,
    df: pl.DataFrame,
    entity_level: EntityLevel,
    entity_filter: set[str] | None,
) -> list[dict[str, object]]:
    entity_expr = entity_identifier_expr(df, entity_level)
    if entity_expr is None:
        return []

    work = df.with_columns(entity_expr.alias("entity")).drop_nulls("entity")
    if entity_filter is not None:
        work = work.filter(pl.col("entity").is_in(entity_filter))
    if work.is_empty():
        return []

    sequence_expr = sequence_expression(work, entity_level)
    base_exprs = [
        sequence_expr.drop_nulls().first().alias("sequence"),
        pl.col("entity").count().alias("rows"),
    ]
    for candidate in (
        "Q.Value",
        "PEP",
        "PG.Q.Value",
        "Protein.Q.Value",
        "RT",
        "Predicted.RT",
        "Best.Fr.Mz.Delta",
        "Ms1.Apex.Mz.Delta",
        "Lib.Q.Value",
        "Lib.Peptidoform.Q.Value",
        "Lib.PG.Q.Value",
        "Evidence",
        "Mass.Evidence",
        "Empirical.Quality",
        "Quantity.Quality",
        "Ms1.Profile.Corr",
        "Precursor.Charge",
    ):
        if candidate in work.columns:
            base_exprs.append(pl.col(candidate).cast(pl.Float64, strict=False).median().alias(candidate))
    if "Precursor.Normalised" in work.columns:
        base_exprs.append(pl.col("Precursor.Normalised").cast(pl.Float64, strict=False).median().alias("abundance"))
    if "nterm_modif" in work.columns:
        base_exprs.append(pl.col("nterm_modif").cast(pl.Utf8).drop_nulls().first().alias("nterm_modif"))
    elif entity_level == "nterm":
        base_exprs.append(pl.col("entity").str.split_exact("_", 1).struct.field("field_0").alias("nterm_modif"))
    if "specificity" in work.columns:
        base_exprs.append(pl.col("specificity").cast(pl.Utf8).drop_nulls().first().alias("specificity"))
    if "protein" in work.columns:
        base_exprs.append(pl.col("protein").cast(pl.Utf8).drop_nulls().first().alias("protein"))
    elif "Protein.Ids" in work.columns:
        base_exprs.append(pl.col("Protein.Ids").cast(pl.Utf8).drop_nulls().first().alias("protein"))

    aggregated = work.group_by("entity").agg(base_exprs).with_columns(pl.lit(search).alias("search"))
    rows = []
    for row in aggregated.to_dicts():
        sequence = str(row.get("sequence") or "")
        row["length"] = len(sequence) if sequence else None
        row["nterm_residue"] = sequence[:1] or None
        row["cterm_residue"] = sequence[-1:] or None
        row["nterm_motif3"] = sequence[:3] or None
        row["cterm_motif3"] = sequence[-3:] or None
        row["gravy"] = gravy(sequence) if sequence else None
        if row.get("RT") is not None and row.get("Predicted.RT") is not None:
            row["Abs.RT.Delta"] = abs(float(row["RT"]) - float(row["Predicted.RT"]))
        if row.get("Best.Fr.Mz.Delta") is not None:
            row["Abs.Best.Fr.Mz.Delta"] = abs(float(row["Best.Fr.Mz.Delta"]))
        if row.get("Ms1.Apex.Mz.Delta") is not None:
            row["Abs.Ms1.Apex.Mz.Delta"] = abs(float(row["Ms1.Apex.Mz.Delta"]))
        for aa in AA_ALPHABET:
            row[f"aa_{aa}"] = sequence.count(aa) / len(sequence) if sequence else None
        rows.append(row)
    return rows


def tables_for_level(
    raw_tables: dict[str, pl.DataFrame] | None,
    adapted_tables: dict[str, pl.DataFrame] | None,
    termini_tables: dict[str, pl.DataFrame] | None,
    entity_level: EntityLevel,
) -> dict[str, pl.DataFrame]:
    if entity_level == "nterm":
        return termini_tables or adapted_tables or raw_tables or {}
    if entity_level == "peptide":
        return raw_tables or adapted_tables or termini_tables or {}
    if entity_level == "peptidoform":
        return raw_tables or {}
    return raw_tables or adapted_tables or termini_tables or {}


def sequence_expression(df: pl.DataFrame, entity_level: EntityLevel) -> pl.Expr:
    if entity_level == "nterm":
        if "peptide" in df.columns:
            return pl.col("peptide").cast(pl.Utf8)
        if "Stripped.Sequence" in df.columns:
            return pl.col("Stripped.Sequence").cast(pl.Utf8)
        return pl.col("entity").str.replace(r"^[^_]+_", "")
    if entity_level == "peptide":
        return pl.col("entity")
    if entity_level == "peptidoform":
        if "Stripped.Sequence" in df.columns:
            return pl.col("Stripped.Sequence").cast(pl.Utf8)
        return pl.col("entity").map_elements(strip_modifications, return_dtype=pl.Utf8)
    return pl.lit(None, dtype=pl.Utf8)


def first_existing_column(df: pl.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def entity_identifier_expr(df: pl.DataFrame, entity_level: EntityLevel) -> pl.Expr | None:
    col = first_existing_column(df, ENTITY_COLUMNS[entity_level])
    if col is not None:
        return pl.col(col).cast(pl.Utf8)
    if entity_level == "nterm" and {"Modified.Sequence", "Stripped.Sequence"}.issubset(df.columns):
        return (
            pl.col("Modified.Sequence")
            .cast(pl.Utf8)
            .map_elements(infer_nterm_modification, return_dtype=pl.Utf8)
            + pl.lit("_")
            + pl.col("Stripped.Sequence").cast(pl.Utf8)
        )
    return None


def infer_nterm_modification(modified_sequence: str | None) -> str:
    """Infer a compact N-term modification label from DIA-NN Modified.Sequence."""

    if not modified_sequence or not modified_sequence.startswith("("):
        return "n"
    match = re.match(r"^\(([^()]*)\)", modified_sequence)
    if not match:
        return "n"
    token = match.group(1)
    if any(marker in token for marker in ("UniMod:36", "UniMod:510", "28.031", "34.063", "36.075")):
        return "Dimethyl"
    if any(marker in token for marker in ("UniMod:737", "UniMod:738", "UniMod:2016", "UniMod:259")):
        return "TMT"
    if any(marker in token for marker in ("UniMod:1", "42.010")):
        return "Acetyl"
    if any(marker in token for marker in ("UniMod:26", "UniMod:27", "89.030")):
        return "2PCA"
    if "UniMod:35" in token or "15.994" in token:
        return "Oxidation"
    return token or "n"


def strip_modifications(sequence: str | None) -> str | None:
    if sequence is None:
        return None
    return re.sub(r"\([^()]*\)", "", sequence)


def empty_intersection_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "intersection_key": pl.Utf8,
            "present_searches": pl.Utf8,
            "absent_searches": pl.Utf8,
            "count": pl.Int64,
            "n_searches": pl.Int64,
        }
    )


def empty_feature_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "entity": pl.Utf8,
            "sequence": pl.Utf8,
            "search": pl.Utf8,
            "rows": pl.Int64,
            "length": pl.Int64,
            "gravy": pl.Float64,
            "nterm_residue": pl.Utf8,
            "cterm_residue": pl.Utf8,
            "nterm_motif3": pl.Utf8,
            "cterm_motif3": pl.Utf8,
        }
    )


def gravy(sequence: str) -> float | None:
    """Return Kyte-Doolittle GRAVY hydrophobicity for an amino-acid sequence."""

    scale = {
        "A": 1.8,
        "C": 2.5,
        "D": -3.5,
        "E": -3.5,
        "F": 2.8,
        "G": -0.4,
        "H": -3.2,
        "I": 4.5,
        "K": -3.9,
        "L": 3.8,
        "M": 1.9,
        "N": -3.5,
        "P": -1.6,
        "Q": -3.5,
        "R": -4.5,
        "S": -0.8,
        "T": -0.7,
        "V": 4.2,
        "W": -0.9,
        "Y": -1.3,
    }
    values = [scale[aa] for aa in sequence if aa in scale]
    if not values:
        return None
    return sum(values) / len(values)
